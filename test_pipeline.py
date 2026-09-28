import tempfile
import unittest
from pathlib import Path
import numpy as np
from audio_pipeline import Segmenter, Transcript, to_16k


class PipelineTests(unittest.TestCase):
    def test_silence_not_transcribed(self):
        segmenter = Segmenter(16000)
        for _ in range(100):
            self.assertIsNone(segmenter.feed(np.zeros(1600, np.float32)))
        self.assertIsNone(segmenter.flush())

    def test_speech_flushes_after_pause(self):
        segmenter = Segmenter(16000)
        for _ in range(5):
            self.assertIsNone(segmenter.feed(np.full(1600, .1, np.float32)))
        chunks = [segmenter.feed(np.zeros(1600, np.float32)) for _ in range(8)]
        self.assertEqual(len(chunks[-1]), 20800)
        self.assertIsNone(segmenter.flush())

    def test_stop_retains_tail_and_maximum_splits(self):
        segmenter = Segmenter(16000, maximum=1)
        chunks = [segmenter.feed(np.full(1600, .1, np.float32)) for _ in range(13)]
        self.assertEqual(len(chunks[9]), 16000)
        self.assertEqual(len(segmenter.flush()), 4800)

    def test_resampling_preserves_duration(self):
        self.assertEqual(len(to_16k(np.zeros(44100, np.float32), 44100)), 16000)

    def test_transcript_is_saved_immediately_and_unique(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Transcript(folder, 'funasr'), Transcript(folder, 'funasr')
            try:
                first.append('你好，语音识别。')
                self.assertIn('你好，语音识别。', first.path.read_text(encoding='utf-8'))
                self.assertNotEqual(first.path, second.path)
            finally:
                first.close()
                second.close()


if __name__ == '__main__':
    unittest.main()
