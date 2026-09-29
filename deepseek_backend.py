"""Optional HTTPS text proofreading. No key, response body or reasoning is logged."""
from dataclasses import dataclass, field
import json
import time
import uuid
import urllib.request
import urllib.error
from settings import LocalLLMConfig
from qwen_backend import QwenLocalBackend
from llm_protocol import (BackendResult, PROMPT_VERSION, build_messages,
                          translate_focused_output, validate_segment_boundary)
from patches import PatchRejected


@dataclass(frozen=True)
class DeepSeekConfig(LocalLLMConfig):
    api_model: str = 'deepseek-flash'
    api_key: str = field(default='', repr=False)
    thinking: bool = True
    max_new_tokens: int = 8192

    def __post_init__(self):
        super().__post_init__()
        if not self.api_model.strip() or len(self.api_model)>128:
            raise ValueError('Invalid DeepSeek model ID')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DeepSeekBackend(QwenLocalBackend):
    source = 'ONLINE_LLM'

    @property
    def model_id(self):
        return self.config.api_model

    def load(self):
        if not self.config.api_key.strip():
            raise ValueError('请在 DeepSeek 设置中输入 API Key（仅保存在内存）')
        if not self.ready:
            self.ready = True
            self.load_count += 1
        self.status('已配置 · 首次转录时连接 DeepSeek')

    def _process_once(self, context, batch_deadline=None):
        start=time.perf_counter()
        metadata=dict(request_id=uuid.uuid4().hex, model_id=self.model_id,
            prompt_version=PROMPT_VERSION, base_revision=context.snapshot.revision,
            started_monotonic=start, input_tokens=0, output_tokens=0)
        try:
            if not self.ready or not self.enabled or self.cancel.is_set():
                raise RuntimeError('API_DISABLED')
            timeout=min(120, self.config.timeout_seconds, (batch_deadline or start+120)-start)
            if timeout<=0:raise TimeoutError()
            payload=dict(model=self.model_id,messages=build_messages(context,self.config.protocol),
                         stream=False,max_tokens=self.config.max_new_tokens,
                         thinking=dict(type='enabled' if self.config.thinking else 'disabled'))
            request=urllib.request.Request('https://api.deepseek.com/chat/completions',
                data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
                headers={'Authorization':'Bearer '+self.config.api_key.strip(),'Content-Type':'application/json'},method='POST')
            self.status('DeepSeek 校对中 · 正在发送转录文字')
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
            patch=translate_focused_output(content,context.snapshot,None if context.punctuation_only else context.evidence)
            validate_segment_boundary(patch,context)
            usage=data.get('usage',{})
            metadata.update(input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0))
            return BackendResult(patch,dict(metadata,latency=time.perf_counter()-start))
        except Exception as exc:
            # Never include str(exc), HTTP bodies, request headers or API keys.
            detail=(exc.code if isinstance(exc,PatchRejected) else
                    'HTTP_'+str(exc.code) if isinstance(exc,urllib.error.HTTPError) else type(exc).__name__)
            return BackendResult(None,dict(metadata,detail=detail,latency=time.perf_counter()-start),'API_ERROR')
        finally:
            self.status('已配置' if self.enabled else 'Disabled')
