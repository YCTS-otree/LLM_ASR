"""Validate actual inference, quantized layers and GPU activation dtype."""
import argparse
import gc
import json
import time
import soundfile as sf
from engines import Engine, ROOT, MODELS
from audio_pipeline import to_16k, Transcript
from evidence import AudioSegment
from version import __version__


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    audio, rate = sf.read(MODELS / 'paraformer' / 'example' / 'asr_example.wav', dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = to_16k(audio, rate)
    modes = [('cpu', 'fp32'), ('cpu', 'int8')]
    if not args.cpu_only:
        modes += [('cuda', 'fp32'), ('cuda', 'fp16'), ('cuda', 'bf16')]
    results = []
    for device, precision in modes:
        engine = Engine(device, precision)
        observed = []
        handle = engine.model.model.encoder.encoders[0].feed_forward.w_1.register_forward_hook(
            lambda module, inputs, output: observed.append(str(output.dtype)))
        start = time.perf_counter()
        evidence = engine.transcribe(AudioSegment.from_audio(audio))
        text = evidence.best_text
        assert len(evidence.nbest) == engine.config.nbest
        assert text == evidence.nbest[0].text
        assert all(isinstance(h.score, float) for h in evidence.nbest)
        assert evidence.timestamps
        handle.remove()
        assert text and '语音识别模型' in text, (device, precision, text)
        if precision in ('fp16', 'bf16'):
            assert str({'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]) in observed, observed
        transcript = Transcript(ROOT / 'transcripts' / 'verification', device + '_' + precision)
        try:
            transcript.append(text)
            assert text in transcript.path.read_text(encoding='utf-8')
        finally:
            transcript.close()
        result = dict(version=__version__, device=engine.device_name, precision=precision, parameters=engine.parameter_count,
                      quantized_layers=engine.quantized_layers, observed_dtypes=observed,
                      seconds=round(time.perf_counter() - start, 3), text=text,
                      nbest=[vars(h) for h in evidence.nbest], timestamps=evidence.timestamps,
                      beam_size=evidence.beam_size)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        engine.close()
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'logs' / ('verification_cpu.json' if args.cpu_only else 'verification.json')).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
