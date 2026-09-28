"""Capture a short local WAV and verify the microphone capture path."""
from datetime import datetime
import threading
import time
import sounddevice as sd
import soundfile as sf
from engines import ROOT
from audio_pipeline import Capture


def main():
    print(sd.query_devices())
    stop = threading.Event()
    path = ROOT / 'recordings' / ('microphone_check_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.wav')
    levels = []
    capture = Capture(None, str(path), stop, .008, levels.append, print)
    capture.start()
    time.sleep(3)
    stop.set()
    capture.join(timeout=10)
    assert not capture.is_alive(), 'Microphone did not stop'
    assert capture.error is None, capture.error
    info = sf.info(path)
    assert info.frames > 0, 'No audio captured'
    print(f'Capture OK: {info.duration:.2f}s, {info.samplerate}Hz, peak RMS {max(levels, default=0):.5f}')
    print(path)


if __name__ == '__main__':
    main()
