"""Small explicit application configuration; no LLM runtime dependency."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ASRConfig:
    beam_size: int = 3
    nbest: int = 3

    def __post_init__(self):
        if type(self.beam_size) is not int or type(self.nbest) is not int:
            raise ValueError('beam_size and nbest must be integers')
        if not 1 <= self.nbest <= self.beam_size <= 32:
            raise ValueError('Require 1 <= nbest <= beam_size <= 32')

    @classmethod
    def from_environment(cls):
        return cls(int(os.getenv('ASR_BEAM_SIZE', '3')), int(os.getenv('ASR_NBEST', '3')))


@dataclass(frozen=True)
class LocalLLMConfig:
    model_path: str = ''
    dtype: str = 'bf16'
    protocol: str = 'anchored'
    max_new_tokens: int = 1024
    max_input_tokens: int = 8192
    timeout_seconds: float = 600.0
    debounce_seconds: float = 0.35
    model_size: str = '2b'
    thinking: bool = False
    target_chars: int = 128
    retry_unchanged: bool = True

    def __post_init__(self):
        from qwen_spec import MODEL_SPECS
        if type(self.target_chars) is not int or not 64 <= self.target_chars <= 2048:
            raise ValueError('Invalid target size')
        if self.model_size not in MODEL_SPECS:
            raise ValueError('Invalid Qwen model size')
        if self.dtype not in ('bf16', 'fp16', 'int4') or self.protocol not in ('relative', 'anchored'):
            raise ValueError('Invalid Qwen dtype/protocol')
        if not 16 <= self.max_new_tokens <= 8192 or not 512 <= self.max_input_tokens <= 16384:
            raise ValueError('Unsafe token limit')
        if not 1 <= self.timeout_seconds <= 1800 or not 0 <= self.debounce_seconds <= 2:
            raise ValueError('Invalid time limit')

    @classmethod
    def from_environment(cls):
        from qwen_spec import MODEL_SPECS
        size = os.getenv('LOCAL_LLM_MODEL_SIZE', '2b').lower()
        if size not in MODEL_SPECS:
            raise ValueError('Invalid Qwen model size')
        return cls(os.getenv('LOCAL_LLM_MODEL_PATH', str(MODEL_SPECS[size]['path'])),
                   os.getenv('LOCAL_LLM_DTYPE', 'bf16'), os.getenv('LOCAL_LLM_PROTOCOL', 'anchored'), model_size=size)
