import tempfile
import logging
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf
from app import Session
from evidence import ASREvidence
from llm_protocol import BackendResult
from settings import ASRConfig, LocalLLMConfig
from correction_models import CorrectionModels


class ComparisonTests(unittest.TestCase):
    def run_session(self, fail_right=False):
        barrier=threading.Barrier(2)
        class Backend:
            enabled=ready=True
            source='LOCAL_LLM'
            def __init__(self,size,mark):
                self.config=SimpleNamespace(model_size=size,debounce_seconds=0)
                self.mark=mark
            def process(self,context):
                barrier.wait(timeout=5)  # Both workers must run, not a sequential fake comparison.
                if fail_right and self.mark=='？':
                    return BackendResult(None,{'model_id':'right'},'RUNTIME_ERROR')
                s=context.snapshot
                return BackendResult(dict(base_revision=s.revision,window_start=s.window_start,
                    window_end=s.window_end,operations=[dict(op='insert',start=s.window_end,end=s.window_end,text=self.mark)]),
                    {'model_id':self.config.model_size})
        calls=[]
        class Engine:
            device='cpu';description='test ASR'
            def transcribe(self,chunk):
                calls.append(chunk.segment_id)
                return ASREvidence(chunk.segment_id,chunk.start_time,chunk.end_time,'你好',end_reason=chunk.end_reason)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.wav';sf.write(source,np.full(16000,.1),16000)
            group=SimpleNamespace(members=(Backend('2b','。'),Backend('0.8b','？')),enabled=True,ready=True)
            host=SimpleNamespace(load=lambda *args:Engine())
            session=Session('cpu','fp32',None,folder,.008,host,ASRConfig(),group,str(source))
            errors=[];session.error.connect(errors.append)
            try:
                with patch('app.ROOT',Path(folder)),patch('app.configure_logging',return_value=logging.getLogger('test')),patch('sounddevice.InputStream',side_effect=AssertionError('mic opened')):
                    session.run()
                self.assertEqual(errors,[])
                self.assertEqual(len(calls),1)
                self.assertEqual(session.store.canonical_text,'你好。')
                self.assertEqual(session.comparison_store.canonical_text,'你好' if fail_right else '你好？')
                for s in (session.store,session.comparison_store):
                    self.assertEqual(s.segments[0].segment_id,calls[0])
                    self.assertEqual(s.segments[0].raw_asr_text,'你好')
                    self.assertEqual(Path(s.path).with_suffix('.txt').read_text(encoding='utf-8'),s.canonical_text)
                    self.assertIsNotNone(s.last_event('LLM_REQUEST'))
                    self.assertIsNone(s.last_event('nonexistent'))
                self.assertNotEqual(session.store.path,session.comparison_store.path)
                self.assertEqual(session.store.evidence(calls[0]),session.comparison_store.evidence(calls[0]))
            finally:session.retire()

    def test_same_asr_independent_concurrent_correction_and_exports(self):
        self.run_session()

    def test_one_model_failure_does_not_disable_other_branch(self):
        self.run_session(fail_right=True)

    def test_model_identity_and_cancel_all(self):
        group=CorrectionModels(LocalLLMConfig(dtype='int4'),mode='both')
        self.assertEqual([m.config.model_size for m in group.members],['2b','0.8b'])
        self.assertIn('Qwen3.5-0.8B',group.members[1].config.model_path)
        group.cancel.set()
        self.assertTrue(all(m.cancel.is_set() for m in group.members))
        group.set_enabled(False)
        self.assertFalse(group.enabled)
        group.close()

    def test_failed_model_load_does_not_skip_other_model(self):
        group=CorrectionModels(LocalLLMConfig(),mode='both')
        with patch.object(group.members[0],'load',side_effect=RuntimeError('OOM')),patch.object(group.members[1],'load') as other:
            with self.assertRaisesRegex(RuntimeError,'OOM'):group.load()
            other.assert_called_once()
        group.close()
