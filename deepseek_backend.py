"""Optional HTTPS text proofreading. No key, response body or reasoning is logged."""
from dataclasses import dataclass, field
import json
import time
import uuid
import logging
import urllib.request
import urllib.error
from pathlib import Path
from settings import LocalLLMConfig
from qwen_backend import QwenLocalBackend
from llm_protocol import (BackendResult, PROMPT_VERSION, build_messages,
                          translate_focused_output, validate_segment_boundary)
from patches import PatchRejected

log = logging.getLogger('meeting_asr')
HTTP_HINTS = {
    400: '请求格式错误，请检查模型和请求配置',
    401: 'API密钥认证失败，请检查配置后重新Load model',
    402: 'API账户余额不足，请在DeepSeek开放平台检查余额并充值后重试',
    422: '请求参数无效，请检查模型支持的参数',
    429: '请求速率达到上限，请稍后重试',
    500: 'DeepSeek服务器故障，请稍后重试',
    503: 'DeepSeek服务器繁忙，请稍后重试',
}


@dataclass(frozen=True)
class DeepSeekConfig(LocalLLMConfig):
    api_model: str = 'deepseek-flash'
    api_key: str = field(default='', repr=False)
    api_key_file: str = 'DEEPSEEK.key'
    thinking: bool = True
    max_new_tokens: int = 8192
    target_chars: int = 512
    retry_unchanged: bool = False
    reasoning_effort: str = 'high'
    output_token_limit: int = 8192
    request_timeout: float = 180.0
    temperature: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.reasoning_effort not in ('low','high','max'):
            raise ValueError('Invalid reasoning effort')
        if type(self.output_token_limit) is not int or not 256 <= self.output_token_limit <= 32768:
            raise ValueError('Invalid output token limit')
        if not 5 <= self.request_timeout <= 600 or not 0 <= self.temperature <= 2:
            raise ValueError('Invalid API generation settings')
        if not self.api_model.strip() or len(self.api_model)>128:
            raise ValueError('Invalid DeepSeek model ID')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DeepSeekBackend(QwenLocalBackend):
    source = 'ONLINE_LLM'

    def __init__(self, config, status=lambda state: None):
        super().__init__(config,status)
        self._api_key = ''

    @property
    def model_id(self):
        return self.config.api_model

    def load(self):
        # Resolve only at explicit Load model, never at import/UI construction.
        # Do not put file contents in configuration repr, exceptions or status.
        self._api_key = ''
        self.ready = False
        value=self.config.api_key.strip()
        if not value:
            try:
                with Path(self.config.api_key_file).open('r',encoding='utf-8-sig') as stream:
                    value=stream.read(4097)
                if len(value)>4096:
                    raise ValueError('DEEPSEEK.key 文件过长，请仅保存密钥文本。')
                value=value.strip()
            except (OSError,UnicodeError):
                raise ValueError('无法读取 DEEPSEEK.key，请在当前运行目录放置 UTF-8 密钥文本，或在设置中输入。') from None
        if not value or len(value)>4096 or any(c.isspace() or not c.isascii() or not c.isprintable() for c in value):
            raise ValueError('DeepSeek 密钥为空或格式无效：应为单个密钥文本，可带首尾换行。')
        self._api_key=value
        if not self.ready:
            self.ready = True
            self.load_count += 1
        self.status('已配置 · 首次转录时连接 DeepSeek')

    def _process_once(self, context, batch_deadline=None):
        start=time.perf_counter()
        metadata=dict(request_id=uuid.uuid4().hex, model_id=self.model_id,
            prompt_version=PROMPT_VERSION, base_revision=context.snapshot.revision,
            punctuation_strategy=context.punctuation_mode,
            started_monotonic=start, input_tokens=0, output_tokens=0)
        try:
            if not self.ready or not self.enabled or self.cancel.is_set():
                raise RuntimeError('API_DISABLED')
            timeout=min(self.config.request_timeout, self.config.timeout_seconds, (batch_deadline or start+self.config.request_timeout)-start)
            if timeout<=0:raise TimeoutError()
            payload=dict(model=self.model_id,messages=build_messages(context,self.config.protocol),
                         stream=False,max_tokens=self.config.output_token_limit,
                         thinking=dict(type='enabled' if self.config.thinking else 'disabled'))
            if self.config.thinking:
                payload['reasoning_effort']=self.config.reasoning_effort
            else:
                payload['temperature']=self.config.temperature
            metadata.update(reasoning_effort=self.config.reasoning_effort if self.config.thinking else 'disabled',
                max_tokens=self.config.output_token_limit,target_chars=self.config.target_chars,request_timeout=timeout)
            request=urllib.request.Request('https://api.deepseek.com/chat/completions',
                data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
                headers={'Authorization':'Bearer '+self._api_key,'Content-Type':'application/json'},method='POST')
            self.status('DeepSeek 校对中 · 正在发送转录文字')
            log.info('DeepSeek 请求开始 id=%s target_chars=%s effort=%s max_tokens=%s timeout=%.1fs',
                     metadata['request_id'],len(context.snapshot.writable_text),metadata['reasoning_effort'],
                     self.config.output_token_limit,timeout)
            chunks=[]
            received=0
            request_deadline=start+timeout
            with urllib.request.build_opener(NoRedirect()).open(request,timeout=timeout) as response:
                while True:
                    if self.cancel.is_set() or not self.enabled:raise RuntimeError('API_CANCELLED')
                    if time.perf_counter()>request_deadline:raise TimeoutError()
                    chunk=response.read1(65536)
                    if not chunk:break
                    received+=len(chunk)
                    if received>2_000_000:raise ValueError('RESPONSE_TOO_LARGE')
                    chunks.append(chunk)
            if time.perf_counter()>request_deadline:raise TimeoutError()
            data=json.loads(b''.join(chunks))
            choice=data['choices'][0]
            if choice.get('finish_reason')!='stop':raise PatchRejected('OUTPUT_INCOMPLETE')
            if not self.enabled or self.cancel.is_set():raise RuntimeError('API_CANCELLED')
            content=choice['message']['content']
            # Some models wrap their final JSON in a Markdown fence.
            if isinstance(content,str) and content.strip().startswith('```json') and content.strip().endswith('```'):
                content=content.strip()[7:-3].strip()
            patch=translate_focused_output(content,context.snapshot,None if context.punctuation_only else context.evidence,normalize_numbers=True)
            from llm_protocol import strict_json
            from number_formatting import normalize_written_numbers
            model_text=strict_json(content)['text']
            metadata['written_number_normalization']=model_text!=normalize_written_numbers(model_text)
            validate_segment_boundary(patch,context)
            usage=data.get('usage',{})
            metadata.update(input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0))
            metadata['reasoning_tokens']=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
            log.info('DeepSeek 请求完成 id=%s latency=%.2fs input_tokens=%s output_tokens=%s reasoning_tokens=%s',
                     metadata['request_id'],time.perf_counter()-start,metadata['input_tokens'],
                     metadata['output_tokens'],metadata['reasoning_tokens'])
            return BackendResult(patch,dict(metadata,latency=time.perf_counter()-start))
        except Exception as exc:
            # Never include str(exc), HTTP bodies, request headers or API keys.
            detail=(exc.code if isinstance(exc,PatchRejected) else
                    'HTTP_'+str(exc.code) if isinstance(exc,urllib.error.HTTPError) else type(exc).__name__)
            hint=HTTP_HINTS.get(exc.code,'HTTP请求失败') if isinstance(exc,urllib.error.HTTPError) else '校对请求未完成，保留原文'
            metadata['error_hint']=hint
            log.warning('DeepSeek 请求失败 id=%s code=%s latency=%.2fs %s',
                        metadata['request_id'],detail,time.perf_counter()-start,hint)
            return BackendResult(None,dict(metadata,detail=detail,latency=time.perf_counter()-start),'API_ERROR')
        finally:
            self.status('已配置' if self.enabled else 'Disabled')

    def close(self):
        super().close()
        self._api_key = ''
