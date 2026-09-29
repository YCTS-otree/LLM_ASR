"""Local Paraformer-large inference with explicit device/precision policies."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import os
import time
import logging
from settings import ASRConfig
from evidence import AudioSegment
from nbest_adapter import ParaformerAdapter, make_evidence

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
MODEL_ID = 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
os.environ.setdefault('MODELSCOPE_CACHE', str(MODELS / 'cache'))


def resolve_runtime(device, precision, cuda_available, bf16_supported=False):
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('未知计算设备')
    if precision not in ('fp32', 'fp16', 'bf16', 'int8'):
        raise ValueError('不支持此精度；RTX 4060 不支持原生 FP6/MXFP6。')
    if precision == 'int8':
        if device == 'cuda':
            raise ValueError('INT8 动态量化使用 CPU。请选择自动或 CPU；GPU 请用 FP16/BF16。')
        return 'cpu'
    resolved = ('cuda' if cuda_available else 'cpu') if device == 'auto' else device
    if resolved == 'cuda' and not cuda_available:
        raise RuntimeError('CUDA 不可用。请运行 install_cuda.ps1 安装 GPU 版 PyTorch，然后重启应用。')
    if precision in ('fp16', 'bf16') and resolved != 'cuda':
        raise ValueError('FP16/BF16 混合精度需要 NVIDIA GPU；CPU 请使用 FP32 或 INT8。')
    if precision == 'bf16' and not bf16_supported:
        raise ValueError('当前 NVIDIA GPU 不支持 BF16。请选择 FP16。')
    return resolved


class Engine:
    def __init__(self, device='auto', precision='fp16', config=None):
        import torch
        from funasr import AutoModel
        available = torch.cuda.is_available()
        self.device = resolve_runtime(device, precision, available,
                                      available and torch.cuda.is_bf16_supported())
        self.precision = precision
        self.config = config or ASRConfig()
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        path = MODELS / 'paraformer'
        if not (path / 'model.pt').exists():
            raise FileNotFoundError('缺少 Paraformer-large 模型，请运行 download_models.py（国内 ModelScope 源）。')
        self.model = AutoModel(model=str(path), device=self.device,
                               disable_update=True, disable_pbar=True,
                               trust_remote_code=False)
        self.parameter_count = sum(p.numel() for p in self.model.model.parameters())
        self.quantized_layers = 0
        if precision in ('fp16', 'bf16'):
            from precision import Float32Module, protect_residuals
            self.model.model.predictor = Float32Module(self.model.model.predictor)
            protect_residuals(self.model.model)
        if precision == 'int8':
            from torch.ao.quantization import quantize_dynamic
            from torch.ao.nn.quantized.dynamic import Linear
            self.model.model = quantize_dynamic(self.model.model, {torch.nn.Linear},
                                                dtype=torch.qint8, inplace=False)
            self.quantized_layers = sum(isinstance(m, Linear) for m in self.model.model.modules())
            if not self.quantized_layers:
                raise RuntimeError('INT8 量化失败：未转换任何 Linear 层。')
        actual = next(self.model.model.parameters()).device.type
        if actual != self.device:
            raise RuntimeError(f'模型设备不匹配：请求 {self.device}，实际 {actual}')
        self.device_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'
        self.description = f'{self.device_name} · {precision.upper()}'
        self.adapter = ParaformerAdapter(self.model, self.config)
        from baseline_punctuation import BaselinePunctuation
        self.punctuation = BaselinePunctuation(MODELS / 'punctuation')

    def transcribe(self, segment):
        import torch
        if not isinstance(segment, AudioSegment) or segment.audio_rate != 16000:
            raise ValueError('Engine requires an AudioSegment resampled to 16 kHz')
        context = (torch.autocast('cuda', dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[self.precision])
                   if self.precision in ('fp16', 'bf16') else nullcontext())
        # AMP retains FP32 for sensitive operators instead of blindly converting the whole network.
        start = time.perf_counter()
        fallback_reason = None
        actual_precision = self.precision
        try:
            with torch.inference_mode(), context:
                result, hypotheses, warning = self.adapter.generate(segment.audio)
        except IndexError as exc:
            if self.precision not in ('fp16', 'bf16'):
                raise
            # A precision-dependent CIF failure can surface as an IndexError.
            # Retry the SAME audio and weights
            # once without autocast; never silently discard that segment.
            fallback_reason = f'{self.precision}: {type(exc).__name__}: {exc}'
            logging.getLogger('meeting_asr').warning('ASR retry in FP32 segment=%s reason=%s', segment.segment_id, fallback_reason)
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                result, hypotheses, warning = self.adapter.generate(segment.audio)
            actual_precision = 'fp32'
        latency = time.perf_counter() - start
        evidence = make_evidence(segment, result, hypotheses, latency, self.config, warning)
        evidence = replace(evidence, inference_precision=actual_precision, fallback_reason=fallback_reason)
        if getattr(self, 'punctuation', None):
            try:
                evidence = replace(evidence, punctuated_text=self.punctuation.punctuate(evidence.best_text, evidence.end_reason))
            except Exception as exc:
                logging.getLogger('meeting_asr').warning('Punctuation failed: %s', type(exc).__name__)
                evidence = replace(evidence, punctuation_error=type(exc).__name__)
        logging.getLogger('meeting_asr').info(
            'ASR segment=%s duration=%.3f latency=%.3f beam_size=%d nbest=%d returned=%d timestamp=%s',
            segment.segment_id, segment.duration, latency, self.config.beam_size,
            self.config.nbest, len(evidence.nbest), evidence.timestamp_warning or 'ok')
        return evidence

    def close(self):
        self.adapter.close()
        self.adapter = None
        self.model = None
        self.punctuation = None
