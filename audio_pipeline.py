"""Microphone capture and pause-based segmentation, independent of inference."""
from collections import deque
from datetime import datetime
from pathlib import Path
import math
import os
import queue
import threading
import uuid
import logging
from dataclasses import replace
from evidence import AudioSegment
import numpy as np
from scipy.signal import resample_poly


class Segmenter:
    def __init__(self, rate, threshold=0.008, silence=0.8, maximum=15):
        self.rate, self.threshold = rate, threshold
        self.silence_limit, self.maximum = int(silence * rate), int(maximum * rate)
        self.pre = deque(maxlen=3)
        self.parts = []
        self.total = self.quiet = self.voiced = 0
        self.position = 0
        self.start_sample = 0

    def feed(self, block):
        block_start = self.position
        self.position += len(block)
        loud = float(np.sqrt(np.mean(block * block))) >= self.threshold
        if not self.parts:
            if not loud:
                self.pre.append(block)
                return None
            self.parts = list(self.pre)
            self.total = sum(len(x) for x in self.parts)
            self.start_sample = block_start - self.total
            self.pre.clear()
        self.parts.append(block)
        self.total += len(block)
        self.voiced += len(block) if loud else 0
        self.quiet = 0 if loud else self.quiet + len(block)
        if self.quiet >= self.silence_limit or self.total >= self.maximum:
            return self.flush('silence' if self.quiet >= self.silence_limit else 'max_duration')
        return None

    def flush(self, reason='manual_stop'):
        audio = np.concatenate(self.parts) if self.parts and self.voiced >= self.rate * .2 else None
        segment = (AudioSegment(uuid.uuid4().hex, self.start_sample, self.start_sample + self.total,
                                self.rate, reason, audio, self.rate,
                                1000 * self.quiet / self.rate) if audio is not None else None)
        self.parts = []
        self.total = self.quiet = self.voiced = 0
        self.pre.clear()
        return segment


def to_16k(audio, rate):
    divisor = math.gcd(rate, 16000)
    return resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)


class Transcript:
    def __init__(self, folder, kind):
        Path(folder).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path = Path(folder) / f'{stamp}_{uuid.uuid4().hex[:8]}_{kind}.txt'
        self.file = self.path.open('x', encoding='utf-8')
        self.file.write(f'语音转写 | {kind} | {datetime.now():%Y-%m-%d %H:%M:%S}\n\n')
        self.file.flush()

    def append(self, text):
        self.file.write(text + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


class Capture(threading.Thread):
    def __init__(self, device, wav_path, stop, threshold, level, status, event=None, queue_size=32):
        super().__init__(daemon=True)
        self.device, self.wav_path, self.stop = device, wav_path, stop
        self.threshold, self.level, self.status = threshold, level, status
        if queue_size < 1:
            raise ValueError('Audio queue must be bounded')
        self.segments = queue.Queue(maxsize=queue_size)
        self.done = threading.Event()
        self.error = None
        self.end_reason = 'manual_stop'
        self.event = event or (lambda kind, data: None)
        self.dropped = self.overflows = self.captured_samples = 0

    def report(self, kind, data):
        logging.getLogger('meeting_asr').warning('%s %s', kind, data)
        self.event(kind, data)

    def enqueue(self, segment):
        segment = replace(segment, audio=to_16k(segment.audio, segment.audio_rate), audio_rate=16000)
        try:
            self.segments.put_nowait(segment)
        except queue.Full:
            self.dropped += 1
            # Raw audio was written before segmentation. Preserve its exact file
            # range for later recovery, keep recording, never grow RAM unbounded.
            self.report('ASR_QUEUE_DROPPED', {
                'segment_id': segment.segment_id, 'start_sample': segment.start_sample,
                'end_sample': segment.end_sample, 'sample_rate': segment.sample_rate,
                'end_reason': segment.end_reason, 'wav_path': self.wav_path,
                'dropped_total': self.dropped})
            self.status(f'识别积压：{self.dropped} 段未实时转写，音频已写入 WAV，详情见会话事件')

    def run(self):
        import sounddevice as sd
        import soundfile as sf
        try:
            info = sd.query_devices(self.device, 'input')
            rate = int(info['default_samplerate'])
            segmenter = Segmenter(rate, threshold=self.threshold)
            Path(self.wav_path).parent.mkdir(parents=True, exist_ok=True)
            # RF64 is a WAV-family container without the ordinary RIFF 4GB limit.
            with sf.SoundFile(self.wav_path, mode='x', samplerate=rate, channels=1,
                              subtype='PCM_16', format='RF64') as wav:
                with sd.InputStream(device=self.device, samplerate=rate, channels=1, dtype='float32') as stream:
                    self.status('正在录音 · 停顿约 0.8 秒后识别')
                    while not self.stop.is_set():
                        frames, overflow = stream.read(max(1, rate // 10))
                        if overflow:
                            self.overflows += 1
                            self.report('DEVICE_OVERFLOW', {'at_captured_sample': self.captured_samples,
                                                           'sample_rate': rate, 'lost_samples': None,
                                                           'overflow_total': self.overflows})
                            self.status('警告：麦克风缓冲溢出，可能有音频丢失')
                        block = frames[:, 0].copy()
                        wav.write(block)
                        self.captured_samples += len(block)
                        if self.captured_samples % rate < len(block):
                            wav.flush()
                        self.level(float(np.sqrt(np.mean(block * block))))
                        chunk = segmenter.feed(block)
                        if chunk is not None:
                            wav.flush()
                            self.enqueue(chunk)
                    chunk = segmenter.flush(self.end_reason)
                    if chunk is not None:
                        wav.flush()
                        self.enqueue(chunk)
        except Exception as exc:
            self.error = f'录音失败：{exc}'
            self.stop.set()
        finally:
            try:
                self.event('CAPTURE_FINISHED', {'captured_samples': self.captured_samples,
                                              'dropped_segments': self.dropped, 'device_overflows': self.overflows,
                                              'error': self.error})
            finally:
                self.done.set()
