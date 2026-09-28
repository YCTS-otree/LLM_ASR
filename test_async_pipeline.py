import threading
import unittest
from evidence import ASREvidence
from llm_pipeline import MeetingPipeline, MockLLMBackend
from transcript_store import TranscriptStore


class AsyncTests(unittest.TestCase):
    def test_end_to_end_mock_patch_retains_raw_and_history(self):
        store = TranscriptStore()
        pipeline = MeetingPipeline(store)
        try:
            pipeline.accept(ASREvidence('s', 0, 2, '我们我们讨论H170和PCIE，，'))
            pipeline.close()
            self.assertEqual(store.canonical_text, '我们讨论HX170和PCIe，')
            self.assertEqual(store.segments[0].raw_asr_text, '我们我们讨论H170和PCIE，，')
            self.assertEqual(store.revision, 2)
            self.assertEqual(store.history()[-1]['source'], 'MOCK_LLM')
        finally:
            if pipeline.worker.is_alive():
                pipeline.close()
            store.close()

    def test_delayed_llm_does_not_block_asr_and_stale_patch_is_discarded(self):
        started, release = threading.Event(), threading.Event()
        class Delayed:
            def __init__(self):
                self.calls = 0
            def process(self, context):
                self.calls += 1
                if self.calls == 1:
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError('test timed out')
                return MockLLMBackend().process(context)
        store = TranscriptStore()
        pipeline = MeetingPipeline(store, backend=Delayed())
        try:
            pipeline.accept(ASREvidence('a', 0, 1, 'H170'))
            self.assertTrue(started.wait(2))
            pipeline.accept(ASREvidence('b', 1, 2, '接口'))
            pipeline.accept(ASREvidence('c', 2, 3, '测试'))
            self.assertEqual(store.revision, 3)
            self.assertEqual(store.canonical_text, 'H170接口测试')
            self.assertLessEqual(pipeline.worker.requests.qsize(), 1)
            release.set()
            pipeline.close()
            events = store.events()
            self.assertTrue(any(e['type'] == 'PATCH_REJECTED' and e['data']['reason'] == 'STALE_PATCH' for e in events))
            self.assertTrue(any(e['type'] == 'LLM_REQUEST_COALESCED' for e in events))
            self.assertEqual(store.canonical_text, 'HX170接口测试')
            self.assertEqual(store.history()[-1]['data']['base_revision'], 3)
        finally:
            release.set()
            if pipeline.worker.is_alive():
                pipeline.close()
            store.close()


if __name__ == '__main__':
    unittest.main()
