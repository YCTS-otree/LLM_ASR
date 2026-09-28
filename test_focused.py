import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from evidence import ASREvidence
from focused_correction import focused_contexts, FOCUS_CHARS
from llm_pipeline import CorrectionContext
from llm_protocol import BackendResult, translate_focused_output
from patches import PatchRejected, Patch, PatchValidator
from qwen_backend import QwenLocalBackend
from settings import LocalLLMConfig
from transcript_store import TranscriptSnapshot, TranscriptStore
from test_transcript import add


class FocusedTests(unittest.TestCase):
    def context(self, text, start=0):
        return CorrectionContext(TranscriptSnapshot(1,text,start,len(text)), ASREvidence('s',0,1,text,end_reason='max_duration'))

    def test_partition_covers_target_with_absolute_offsets_and_readonly_context(self):
        c=self.context('已冻结。'+'现在我们讨论语音转录的准确性'*9,4)
        parts=focused_contexts(c)
        self.assertEqual(''.join(p.snapshot.writable_text for p in parts),c.snapshot.writable_text)
        for p in parts:
            self.assertLessEqual(len(p.snapshot.writable_text),FOCUS_CHARS)
            self.assertGreaterEqual(p.snapshot.window_start,c.snapshot.window_start)
            self.assertEqual(p.snapshot.window_end,len(p.snapshot.text))
        self.assertTrue(parts[0].readable_right)
        self.assertEqual(parts[-1].evidence.end_reason,'max_duration')

    def test_latin_word_does_not_split_when_boundary_available(self):
        c=self.context('中'*62+'Paraformer模型继续讨论')
        self.assertEqual(focused_contexts(c)[0].snapshot.window_end,62)

    def test_compact_output_cannot_copy_context_or_change_revision(self):
        s=self.context('冻结原文没有句号',2).snapshot
        patch=translate_focused_output(json.dumps(dict(base_revision=1,text='原文没有句号。')),s)
        self.assertEqual(patch['operations'],[dict(op='insert',start=8,end=8,text='。')])
        for text in ['原文没有句号，顺便把前后文整段复制过来','完全换了一大段新内容']:
            with self.assertRaisesRegex(PatchRejected,'CONTENT_DRIFT'):
                translate_focused_output(json.dumps(dict(base_revision=1,text=text)),s)
        with self.assertRaisesRegex(PatchRejected,'REVISION_MISMATCH'):
            translate_focused_output(json.dumps(dict(base_revision=2,text=s.writable_text)),s)

    def test_combination_keeps_good_target_when_another_fails(self):
        c=self.context('中'*160)
        backend=QwenLocalBackend(LocalLLMConfig())
        backend.ready=True
        def result(target, deadline):
            s=target.snapshot
            if s.window_start==64:
                return BackendResult(None,dict(detail='CONTENT_DRIFT'),'INVALID_OUTPUT')
            return BackendResult(translate_focused_output(json.dumps(dict(base_revision=1,text=s.writable_text+'，')),s))
        backend._process_once=Mock(side_effect=result)
        r=backend.process(c)
        self.assertIsNone(r.error)
        self.assertEqual(r.metadata['partial_failures'],1)
        self.assertEqual(len(r.patch['operations']),2)
        PatchValidator().validate(Patch.parse(r.patch),c.snapshot,1,c.snapshot.text,0)

    def test_cancellation_stops_subsequent_targets(self):
        c=self.context('中'*160)
        backend=QwenLocalBackend(LocalLLMConfig())
        backend.ready=True
        def result(target, deadline):
            backend.set_enabled(False)
            return BackendResult(None,dict(detail='cancelled'),'RUNTIME_ERROR')
        backend._process_once=Mock(side_effect=result)
        r=backend.process(c)
        self.assertEqual(backend._process_once.call_count,1)
        self.assertIsNone(r.patch)

    def test_2048_window_persists_and_never_unfreezes_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'session.sqlite3'
            store=TranscriptStore(path,window_chars=2048)
            add(store,'中'*2100)
            self.assertEqual(store.frozen_boundary,52)
            store.close()
            store=TranscriptStore(path)
            self.assertEqual(store.window_chars,2048)
            store.close()
            store=TranscriptStore(path,window_chars=4096)
            add(store,'文')
            self.assertEqual(store.frozen_boundary,52)
            store.close()
        for invalid in [0,255,4097,True,1.5]:
            with self.assertRaises(ValueError): TranscriptStore(window_chars=invalid)


if __name__=='__main__': unittest.main()
