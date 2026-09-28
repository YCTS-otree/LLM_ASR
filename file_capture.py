"""Bounded offline audio reader; source file is read-only, never a microphone."""
from dataclasses import replace
from pathlib import Path
import queue
import threading
import numpy as np
from audio_pipeline import Capture, Segmenter, to_16k


class FileCapture(Capture):
    def __init__(self, source, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.source = str(Path(source).resolve())
        self.abort = threading.Event()
        self.end_reason = 'end_of_file'

    def enqueue(self, segment):
        item = replace(segment, audio=to_16k(segment.audio, segment.audio_rate), audio_rate=16000)
        while not self.abort.is_set():
            try:
                self.segments.put(item, timeout=.1)
                return
            except queue.Full:
                continue  # Offline input can wait; it must not use mic drop policy.

    def run(self):
        import soundfile as sf
        total = 0
        try:
            if Path(self.source) == Path(self.wav_path).resolve():
                raise ValueError('Source and output must differ')
            Path(self.wav_path).parent.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(self.source, mode='r') as source:
                total, rate = len(source), source.samplerate
                segmenter = Segmenter(rate, threshold=self.threshold)
                self.event('FILE_IMPORT_START', dict(source=self.source, frames=total,
                    sample_rate=rate, channels=source.channels))
                with sf.SoundFile(self.wav_path, mode='x', samplerate=rate, channels=1,
                                  subtype='PCM_16', format='RF64') as wav:
                    while not self.stop.is_set() and not self.abort.is_set():
                        frames = source.read(max(1,rate//10), dtype='float32', always_2d=True)
                        if not len(frames): break
                        block = frames.mean(axis=1)
                        wav.write(block)
                        self.captured_samples += len(block)
                        chunk = segmenter.feed(block)
                        if self.captured_samples % rate < len(block):
                            self.level(float(np.sqrt(np.mean(block*block))))
                            self.status(f'正在离线转录 · 已读取 {self.captured_samples/max(1,total):.0%}')
                        if chunk is not None:
                            wav.flush()
                            self.enqueue(chunk)
                    chunk = segmenter.flush(self.end_reason)
                    if chunk is not None and not self.abort.is_set():
                        wav.flush()
                        self.enqueue(chunk)
        except Exception as exc:
            self.error = f'导入音频失败（格式可能不受支持）：{exc}'
            self.stop.set()
        finally:
            try:
                self.event('FILE_IMPORT_FINISHED',dict(source=self.source,total_samples=total,
                    captured_samples=self.captured_samples,cancelled=self.stop.is_set(),error=self.error))
            finally:
                self.done.set()
