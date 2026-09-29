import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from compare_subtitles import read_ass, character_error_rate
from engines import Engine
from evidence import AudioSegment
from settings import ASRConfig


class QualityTests(unittest.TestCase):
    def test_ass_text_commas_and_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reference.ass'
            path.write_text('Dialogue: 0,0:00:00.20,0:00:02.00,Default,,0,0,0,,{\\b1}你好,世界\\N继续\nDialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,后文',encoding='utf-8')
            self.assertEqual(read_ass(path,0,2),'你好,世界 继续')
            with self.assertRaises(ValueError): read_ass(path,0,1)

    def test_cer_counts_omissions_not_punctuation(self):
        self.assertEqual(character_error_rate('你好，世界！','你好世界')['cer'],0)
        r=character_error_rate('甲乙丙丁','丙丁')
        self.assertEqual(r['edit_distance'],2)
        self.assertEqual(r['cer'],.5)

    def test_amp_failure_retries_same_segment_in_fp32_and_audits_precision(self):
        engine=object.__new__(Engine)
        engine.precision='fp16';engine.config=ASRConfig();engine.adapter=Mock()
        engine.adapter.generate.side_effect=[IndexError('zero length CIF'),([{'text':'开头仍然保留'}],(),None)]
        segment=AudioSegment.from_audio(np.zeros(16000,dtype=np.float32))
        with self.assertLogs('meeting_asr',level='WARNING'):
            result=engine.transcribe(segment)
        self.assertEqual(result.best_text,'开头仍然保留')
        self.assertEqual(result.inference_precision,'fp32')
        self.assertIn('fp16',result.fallback_reason)
        self.assertIs(engine.adapter.generate.call_args_list[0].args[0],engine.adapter.generate.call_args_list[1].args[0])

    def test_fp32_failure_propagates_instead_of_looping(self):
        engine=object.__new__(Engine)
        engine.precision='fp32';engine.config=ASRConfig();engine.adapter=Mock()
        engine.adapter.generate.side_effect=IndexError('failure')
        with self.assertRaises(IndexError):engine.transcribe(AudioSegment.from_audio(np.zeros(16000)))
        self.assertEqual(engine.adapter.generate.call_count,1)
