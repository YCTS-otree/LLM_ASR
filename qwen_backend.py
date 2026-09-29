"""Resident local text-only Qwen backend. Has no TranscriptStore reference."""
import logging
import threading
import time
import uuid
import unicodedata
from pathlib import Path
from dataclasses import replace
from llm_protocol import BackendResult, PROMPT_VERSION, build_messages, translate_output, translate_focused_output, validate_segment_boundary
from patches import PatchRejected, Patch, PatchValidator
from focused_correction import focused_contexts
from qwen_spec import MODEL_SPECS

log = logging.getLogger('meeting_asr')


class QwenLocalBackend:
    source = 'LOCAL_LLM'

    @property
    def model_id(self):
        return MODEL_SPECS[self.config.model_size]['model_id']

    def __init__(self, config, status=lambda state: None):
        self.config, self.status = config, status
        self.model = self.tokenizer = None
        self.enabled = True
        self.ready = False
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.load_count = 0
        self.last_metadata = {}

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not enabled:
            self.cancel.set()
        self.status('Ready' if self.enabled and self.ready else ('Loading' if self.enabled else 'Disabled'))

    def load(self):
        with self.lock:
            if self.ready:
                return
            self.status('Loading')
            try:
                path = Path(self.config.model_path or MODEL_SPECS[self.config.model_size]['path']).resolve()
                if not (path / 'config.json').is_file() or not (path / 'model.safetensors.index.json').is_file():
                    raise FileNotFoundError(f'本地 Qwen 模型缺失：{path}；请显式运行 download_qwen.py --download --model {self.config.model_size}')
                import torch
                from transformers import AutoTokenizer, Qwen3_5ForCausalLM
                if not torch.cuda.is_available():
                    raise RuntimeError('Qwen baseline requires CUDA; ASR remains available')
                dtype = torch.bfloat16 if self.config.dtype in ('bf16', 'int4') else torch.float16
                options = dict(local_files_only=True, trust_remote_code=False, use_safetensors=True,
                               dtype=dtype, device_map='cuda:0', attn_implementation='sdpa')
                if self.config.dtype == 'int4':
                    from transformers import BitsAndBytesConfig
                    options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True)
                self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
                self.model, info = Qwen3_5ForCausalLM.from_pretrained(str(path), output_loading_info=True, **options)
                if info.get('missing_keys') or info.get('mismatched_keys'):
                    raise RuntimeError('Qwen text weights did not load completely')
                self.model.eval()
                self.ready = True
                self.load_count += 1
                self.status('Ready' if self.enabled else 'Disabled')
            except Exception:
                self.model = self.tokenizer = None
                self.ready = False
                self.status('Error')
                raise

    def process(self, context):
        if self.enabled and self.ready:
            self.cancel.clear()
        targets=focused_contexts(context)
        if len(targets)==1 and targets[0] is context:
            return self._process_once(context)
        started=time.perf_counter()
        deadline=started+self.config.timeout_seconds
        s=context.snapshot
        combined=dict(base_revision=s.revision,window_start=s.window_start,window_end=s.window_end,operations=[])
        records=[]
        for target in targets:
            if time.perf_counter()>=deadline or not self.enabled or self.cancel.is_set():
                records.append(dict(result='RUNTIME_ERROR',detail='BATCH_CANCELLED_OR_TIMEOUT'))
                break
            result=self._process_once(target,deadline)
            attempts=[]
            if target.focused and (result.error in ('INVALID_OUTPUT','VALIDATION_REJECTED') or (result.patch is not None and not result.patch['operations'] and len(target.snapshot.writable_text)>24)):
                # Quality over latency: one conservative punctuation-only retry.
                # Both attempts use the same immutable snapshot and deadline.
                attempts.append(dict(result.metadata,result=result.error or 'VALID'))
                if time.perf_counter()<deadline and self.enabled and not self.cancel.is_set():
                    retry=self._process_once(replace(target,punctuation_only=True),deadline)
                    attempts.append(dict(retry.metadata,result=retry.error or 'VALID'))
                    if retry.patch is not None or result.patch is None:
                        result=retry
            record=dict(result.metadata,result=result.error or 'VALID',focus_start=target.snapshot.window_start,focus_end=target.snapshot.window_end)
            if attempts:
                record['attempts']=attempts
                record['input_tokens']=sum(a.get('input_tokens',0) for a in attempts)
                record['output_tokens']=sum(a.get('output_tokens',0) for a in attempts)
            if result.patch:
                # Adjacent targets can propose the same punctuation at their
                # shared boundary. Keep an identical operation only once;
                # conflicting proposals still go through the strict validator.
                operations=list(combined['operations'])
                operations.extend(op for op in result.patch['operations'] if op not in operations)
                candidate=dict(combined,operations=operations)
                try:
                    PatchValidator().validate(Patch.parse(candidate),s,s.revision,s.text,s.window_start)
                    combined=candidate
                except PatchRejected as exc:
                    record.update(result='VALIDATION_REJECTED',detail=exc.code)
            records.append(record)
        metadata=dict(records[0] if records else {},request_id=uuid.uuid4().hex,
            model_id=self.model_id,prompt_version=PROMPT_VERSION,
            dtype=self.config.dtype,protocol=self.config.protocol,base_revision=s.revision,
            window_start=s.window_start,window_length=len(s.writable_text),started_monotonic=started,
            latency=time.perf_counter()-started,subrequests=records,focus_count=len(targets),
            input_tokens=sum(r.get('input_tokens',0) for r in records),output_tokens=sum(r.get('output_tokens',0) for r in records),
            partial_failures=sum(r['result']!='VALID' for r in records))
        metadata.pop('result',None)
        revised=s.text
        for op in sorted(combined['operations'],key=lambda item:(item['start'],item['end']),reverse=True):
            revised=revised[:op['start']]+op['text']+revised[op['end']:]
        lexical=lambda text: ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))
        metadata['lexical_changed']=lexical(revised)!=lexical(s.text)
        metadata['attempt_rejections']=sum(a.get('result')!='VALID' for r in records for a in r.get('attempts',[r]))
        if any(r['result']=='VALID' for r in records):
            metadata.pop('detail',None)
        self.last_metadata=metadata
        if combined['operations'] or any(r['result']=='VALID' for r in records):
            return BackendResult(combined,metadata)
        return BackendResult(None,metadata,records[0]['result'] if records else 'RUNTIME_ERROR')

    def _process_once(self, context, batch_deadline=None):
        started = time.perf_counter()
        final_state = 'Ready'
        metadata = dict(request_id=uuid.uuid4().hex, model_id=self.model_id, prompt_version=PROMPT_VERSION,
                        started_monotonic=started,
                        protocol=self.config.protocol, dtype=self.config.dtype, base_revision=context.snapshot.revision,
                        window_start=context.snapshot.window_start, window_length=len(context.snapshot.writable_text),
                        segment_ids=[context.evidence.segment_id], nbest_count=len(context.evidence.nbest),
                        input_tokens=0, output_tokens=0, prefill_latency=None, decode_latency=None)
        try:
            if not self.enabled or not self.ready:
                raise RuntimeError('Local correction disabled or model not ready')
            # One process call per worker; lock also protects close/load across sessions.
            with self.lock:
                import torch
                from transformers import StoppingCriteria, StoppingCriteriaList
                if batch_deadline is None:
                    self.cancel.clear()
                if not self.enabled:
                    raise RuntimeError('Local correction disabled')
                messages = build_messages(context, self.config.protocol)
                thinking=self.config.thinking and not context.punctuation_only
                metadata['thinking']=thinking
                inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=thinking, return_tensors='pt', return_dict=True)
                metadata['input_tokens'] = inputs['input_ids'].shape[-1]
                if metadata['input_tokens'] > self.config.max_input_tokens:
                    raise RuntimeError('Input token budget exceeded; transcript preserved')
                inputs = inputs.to('cuda:0')
                deadline = min(batch_deadline or float('inf'),time.perf_counter() + self.config.timeout_seconds)
                cancel = self.cancel
                class Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return cancel.is_set() or time.perf_counter() >= deadline
                class Timing:
                    def __init__(self): self.calls, self.first_token = 0, None
                    def put(self, value):
                        self.calls += 1
                        if self.calls == 2: self.first_token = time.perf_counter()
                    def end(self): pass
                timing = Timing()
                inference_start = time.perf_counter()
                self.status('Busy')
                with torch.inference_mode():
                    output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                               use_cache=True, streamer=timing, stopping_criteria=StoppingCriteriaList([Stop()]))
                generated = output[0, inputs['input_ids'].shape[-1]:].tolist()
                ended = time.perf_counter()
                metadata.update(output_tokens=len(generated),
                    prefill_latency=(timing.first_token - inference_start if timing.first_token else None),
                    decode_latency=(ended - timing.first_token if timing.first_token else None))
                raw = self.tokenizer.decode(generated, skip_special_tokens=True)
                if thinking:
                    # Only the final answer is parsed or logged, never reasoning.
                    if '</think>' not in raw:
                        raise PatchRejected('INCOMPLETE_THINKING')
                    raw=raw.rsplit('</think>',1)[1].strip()
                del output, inputs
                if self.cancel.is_set() or not self.enabled:
                    raise RuntimeError('Correction cancelled')
                if ended >= deadline:
                    raise TimeoutError('Correction timeout (checked between generation steps)')
                if len(generated) >= self.config.max_new_tokens:
                    raise PatchRejected('OUTPUT_TOKEN_LIMIT')
                # Full prompts/raw output are debug-only, under rotating log limits.
                log.debug('Qwen request=%s messages=%s raw=%s', metadata['request_id'], messages, raw)
                patch = translate_focused_output(raw,context.snapshot,None if context.punctuation_only else context.evidence) if context.focused else translate_output(raw, context.snapshot, self.config.protocol)
                validate_segment_boundary(patch, context)
                metadata['latency'] = time.perf_counter() - started
                self.last_metadata = dict(metadata)
                return BackendResult(patch, metadata)
        except PatchRejected as exc:
            final_state = 'Error'
            metadata.update(latency=time.perf_counter() - started, detail=exc.code)
            return BackendResult(None, metadata, 'VALIDATION_REJECTED' if exc.code in ('HARD_CUT_TERMINATOR', 'ORAL_FILLER_REMOVED') else 'INVALID_OUTPUT')
        except Exception as exc:
            final_state = 'Error'
            metadata.update(latency=time.perf_counter() - started, detail=f'{type(exc).__name__}: {exc}')
            log.warning('Local correction failed request=%s error=%s', metadata['request_id'], metadata['detail'])
            return BackendResult(None, metadata, 'RUNTIME_ERROR')
        finally:
            self.last_metadata = dict(metadata)
            self.status(final_state if self.ready and self.enabled else ('Disabled' if not self.enabled else 'Error'))

    def close(self):
        self.cancel.set()
        with self.lock:
            self.ready = False
            self.model = self.tokenizer = None
        self.status('Disabled')
