import threading
import unittest
from unittest.mock import Mock
import numpy as np
from evidence import AudioSegment, ASREvidence, Hypothesis, StabilityBuffer
from audio_pipeline import Segmenter, Capture
from nbest_adapter import make_evidence, ParaformerAdapter
from settings import ASRConfig
from model_host import ModelHost


class EvidenceTests(unittest.TestCase):
    def test_top_one_score_and_timestamps(self):
        segment = AudioSegment('s', 16000, 32000, 16000, 'silence', np.zeros(16000))
        hypotheses = [Hypothesis(1, 'HX170', -1.5), Hypothesis(2, 'H170', -7.0)]
        ev = make_evidence(segment, [{'text': 'HX170', 'timestamp': [[0, 100]]}, {'text': 'H170'}],
                           hypotheses, .2, ASRConfig())
        self.assertEqual(ev.best_text, 'HX170')
        self.assertEqual(ev.nbest[1].score, -7.0)
        self.assertIsNone(ev.confidence)
        self.assertEqual(ev.timestamps, ((0, .1),))
        self.assertEqual(ev.global_timestamps, ((1, 1.1),))
        self.assertEqual(len(ev.raw_result), 2)

    def test_timestamp_missing_invalid_or_unmatched_does_not_fail_text(self):
        segment = AudioSegment.from_audio(np.zeros(16000))
        for value in [None, 'bad', [[1]], [[-1, 5]], [[20, 10]], [[0, 5000]], [[0, float('nan')]]]:
            ev = make_evidence(segment, [{'text': '原文', 'timestamp': value}], [], 0, ASRConfig())
            self.assertEqual(ev.best_text, '原文')
            self.assertEqual(ev.timestamps, ())
        ev = make_evidence(segment, [{'text': '许多字符', 'timestamp': [[0, 100]]}], [], 0, ASRConfig())
        self.assertEqual(ev.timestamps, ((0, .1),))

    def test_sample_positions_include_preroll_and_gaps(self):
        segmenter = Segmenter(16000, maximum=1)
        for _ in range(10):
            segmenter.feed(np.zeros(1600, np.float32))
        result = None
        for _ in range(7):
            result = segmenter.feed(np.full(1600, .1, np.float32))
        self.assertEqual((result.start_sample, result.end_sample), (11200, 27200))
        self.assertEqual(result.end_reason, 'max_duration')
        for _ in range(2):
            segmenter.feed(np.full(1600, .1, np.float32))
        tail = segmenter.flush('shutdown_flush')
        self.assertEqual((tail.start_sample, tail.end_sample), (27200, 30400))
        self.assertEqual(tail.end_reason, 'shutdown_flush')

    def test_end_reasons(self):
        segmenter = Segmenter(16000)
        for _ in range(3):
            segmenter.feed(np.full(1600, .1, np.float32))
        for _ in range(8):
            result = segmenter.feed(np.zeros(1600, np.float32))
        self.assertEqual(result.end_reason, 'silence')
        for _ in range(3):
            segmenter.feed(np.full(1600, .1, np.float32))
        self.assertEqual(segmenter.flush().end_reason, 'manual_stop')

    def test_stability_preserves_hard_cut_and_order(self):
        buffer = StabilityBuffer()
        first = buffer.push(ASREvidence('a', 0, 15, '尚未', end_reason='max_duration'))
        second = buffer.push(ASREvidence('b', 15, 17, '说完', end_reason='silence'))
        self.assertTrue(first.hard_cut)
        self.assertFalse(first.natural_pause)
        self.assertTrue(second.follows_hard_cut)
        self.assertEqual(second.previous_segment_id, 'a')
        with self.assertRaises(ValueError):
            buffer.push(ASREvidence('b', 15, 17, '重复'))

    def test_full_audio_queue_keeps_recording_and_records_range(self):
        events = []
        stop = threading.Event()
        capture = Capture(0, 'raw.wav', stop, .01, lambda x: None, lambda s: None,
                          lambda kind, data: events.append((kind, data)), queue_size=1)
        capture.enqueue(AudioSegment.from_audio(np.ones(3200, np.float32)))
        dropped = AudioSegment('dropped', 3200, 6400, 16000, 'manual_stop', np.ones(3200, np.float32))
        capture.enqueue(dropped)
        self.assertFalse(stop.is_set())
        self.assertEqual(capture.segments.qsize(), 1)
        self.assertEqual(events[0][0], 'ASR_QUEUE_DROPPED')
        self.assertEqual(events[0][1]['start_sample'], 3200)

    def test_model_resident_until_configuration_change_or_exit(self):
        factory = Mock(side_effect=[Mock(), Mock()])
        host = ModelHost(factory)
        first = host.load('auto', 'fp16', ASRConfig())
        self.assertIs(first, host.load('auto', 'fp16', ASRConfig()))
        self.assertEqual(factory.call_count, 1)
        self.assertIsNot(first, host.load('cpu', 'int8', ASRConfig(1, 1)))
        self.assertEqual(factory.call_count, 2)
        host.close()
        self.assertIsNone(host.engine)

    def test_beam_config_validation(self):
        for args in [(0, 1), (1, 3), (True, 1), (33, 1)]:
            with self.assertRaises(ValueError):
                ASRConfig(*args)

    def test_timestamp_postprocessor_failure_retries_text_only(self):
        adapter = object.__new__(ParaformerAdapter)
        adapter.auto_model = Mock()
        adapter.auto_model.generate.side_effect = [IndexError('alignment mismatch'), [{'text': '原文'}]]
        with self.assertLogs('meeting_asr', level='WARNING'):
            result, hypotheses, warning = adapter.generate(np.zeros(16000))
        self.assertEqual(result[0]['text'], '原文')
        self.assertEqual(warning, 'timestamp_fallback')
        self.assertFalse(adapter.auto_model.generate.call_args.kwargs['pred_timestamp'])


if __name__ == '__main__':
    unittest.main()
