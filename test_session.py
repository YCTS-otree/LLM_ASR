import queue
import tempfile
import threading
import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from app import Session
from evidence import AudioSegment, ASREvidence
from settings import ASRConfig
from llm_protocol import BackendResult


class SessionTests(unittest.TestCase):
    def test_offline_asr_waits_for_each_correction_before_next_segment(self):
        order=[]
        class Reader:
            def __init__(self,*args,**kwargs):
                self.segments=queue.Queue()
                for i in range(2):
                    self.segments.put(AudioSegment(str(i),i*16000,(i+1)*16000,16000,'silence',np.zeros(16000)))
                self.done=threading.Event(); self.done.set()
                self.error=None; self.dropped=self.overflows=0
            def start(self): pass
            def join(self): pass
        class Backend:
            enabled=ready=True
            source='LOCAL_LLM'
            config=SimpleNamespace(model_size='2b',debounce_seconds=0)
            def process(self,context):
                time.sleep(.03)
                order.append('LLM '+context.evidence.segment_id)
                s=context.snapshot
                return BackendResult(dict(base_revision=s.revision,window_start=s.window_start,window_end=s.window_end,operations=[]),
                                     dict(request_id=context.evidence.segment_id))
        engine=Mock(device='cpu',description='test')
        def transcribe(chunk):
            order.append('ASR '+chunk.segment_id)
            return ASREvidence(chunk.segment_id,chunk.start_time,chunk.end_time,'测试原文')
        engine.transcribe.side_effect=transcribe
        host=Mock(); host.load.return_value=engine
        with tempfile.TemporaryDirectory() as folder:
            session=Session('cpu','fp32',None,folder,.008,host,ASRConfig(),Backend(),'example.wav',window_chars=2048)
            with patch('app.FileCapture',Reader),patch('app.configure_logging',return_value=Mock()):
                session.run()
            try:
                self.assertEqual(order,['ASR 0','LLM 0','ASR 1','LLM 1'])
                self.assertEqual(session.store.window_chars,2048)
                requests=[e for e in session.store.events() if e['type']=='LLM_REQUEST']
                self.assertEqual(len(requests),2)
            finally: session.retire()

    def test_failed_segment_is_audited_and_next_segment_still_corrected(self):
        class FakeCapture:
            def __init__(self, *args, **kwargs):
                self.segments = queue.Queue()
                self.segments.put(AudioSegment('failed', 0, 16000, 16000, 'silence', np.zeros(16000)))
                self.segments.put(AudioSegment('ok', 16000, 32000, 16000, 'manual_stop', np.zeros(16000)))
                self.done = threading.Event()
                self.done.set()
                self.error = None
                self.dropped = self.overflows = 0
            def start(self): pass
            def join(self): pass
        engine = Mock(device='cpu', description='test CPU')
        engine.transcribe.side_effect = [RuntimeError('inference failure'),
                                        ASREvidence('ok', 1, 2, 'H170', end_reason='manual_stop')]
        host = Mock()
        host.load.return_value = engine
        with tempfile.TemporaryDirectory() as folder:
            session = Session('cpu', 'fp32', 1, folder, .008, host, ASRConfig())
            statuses, errors = [], []
            session.status.connect(statuses.append)
            session.error.connect(errors.append)
            with patch('app.Capture', FakeCapture), patch('app.configure_logging', return_value=Mock()):
                session.run()
            try:
                self.assertEqual(errors, [])
                self.assertEqual(session.store.canonical_text, 'HX170')
                self.assertEqual(session.store.segments[0].raw_asr_text, 'H170')
                failed = [e for e in session.store.events() if e['type'] == 'ASR_FAILED']
                self.assertEqual(failed[0]['data']['segment_id'], 'failed')
                self.assertEqual(failed[0]['data']['end_sample'], 16000)
                self.assertIn('识别失败 1 段', statuses[-1])
                engine.close.assert_not_called()
                host.close.assert_not_called()
            finally:
                session.retire()


if __name__ == '__main__':
    unittest.main()
