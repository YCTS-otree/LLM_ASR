import json
import tempfile
import threading
import time
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from audio_pipeline import Segmenter
from evidence import ASREvidence, StabilityBuffer
from llm_pipeline import ContextBuilder
from llm_protocol import translate_output, validate_segment_boundary
from patches import RevisionEngine, PatchRejected
from transcript_store import TranscriptStore
from punctuation_cases import CASES
from file_capture import FileCapture


class MinimalPunctuationTests(unittest.TestCase):
    def apply_text(self, segments, new):
        store=TranscriptStore()
        stability=StabilityBuffer()
        for i,(text,reason) in enumerate(segments):
            stable=stability.push(ASREvidence(str(i),i,i+1,text,end_reason=reason,pause_after_ms=800))
            store.append(stable)
        snapshot=store.snapshot()
        raw=json.dumps(dict(base_revision=snapshot.revision,operations=[dict(op='replace',old_text=snapshot.text,new_text=new)]))
        patch=translate_output(raw,snapshot,'anchored')
        RevisionEngine(store).apply(patch,snapshot,'LOCAL_LLM')
        return store,patch,ContextBuilder().build(store,stable)

    def test_all_reference_punctuation_cases_preserve_each_segments_words(self):
        punctuation='，。？！：；,.?!:;'
        for name,segments,expected in CASES:
            with self.subTest(case=name):
                store,patch,context=self.apply_text(segments,expected)
                try:
                    self.assertEqual(store.canonical_text,expected)
                    for s,(raw,_) in zip(store.segments,segments):
                        self.assertEqual(''.join(c for c in s.current_text if c not in punctuation),raw)
                        self.assertEqual(s.raw_asr_text,raw)
                        self.assertEqual(s.asr_evidence_ref,s.segment_id)
                    self.assertTrue(all(op['op']=='insert' for op in patch['operations']))
                    self.assertEqual(context.segments[0]['end_reason'],segments[0][1])
                finally: store.close()

    def test_boundary_punctuation_belongs_to_previous_segment(self):
        store,_,_=self.apply_text([('你好','silence'),('我们开始','silence')],'你好，我们开始。')
        try:
            self.assertEqual([s.current_text for s in store.segments],['你好，','我们开始。'])
            self.assertEqual([(s.start_time,s.end_time) for s in store.segments],[(0,1),(1,2)])
        finally: store.close()

    def test_noop_does_not_increment_revision(self):
        store,patch,_=self.apply_text([('你好。','silence')],'你好。')
        try:
            self.assertEqual(patch['operations'],[])
            self.assertEqual(store.revision,1)
        finally:store.close()

    def test_source_size_overlap_and_frozen_still_rejected(self):
        from transcript_store import TranscriptSnapshot
        for snapshot,ops in [
            (TranscriptSnapshot(1,'a'*200,0,200),[dict(op='replace',old_text='a'*200,new_text='a'*200+'。')]),
            (TranscriptSnapshot(1,'你好我们',0,4),[dict(op='replace',old_text='你好',new_text='你好，'),dict(op='replace',old_text='你好我们',new_text='你好我们。')]),
            (TranscriptSnapshot(1,'冻结你好',2,4),[dict(op='replace',old_text='冻结',new_text='冻结，')])]:
            with self.assertRaises(PatchRejected):
                translate_output(json.dumps(dict(base_revision=1,operations=ops)),snapshot,'anchored')

    def test_pause_uses_trailing_threshold_silence(self):
        s=Segmenter(16000)
        for _ in range(3):s.feed(np.full(1600,.1,np.float32))
        for _ in range(8): item=s.feed(np.zeros(1600,np.float32))
        self.assertEqual(item.end_reason,'silence')
        self.assertEqual(item.pause_after_ms,800)

    def test_spoken_fillers_cannot_be_removed_during_punctuation(self):
        store=TranscriptStore()
        stable=StabilityBuffer().push(ASREvidence('filler',0,1,'嗯啊啊呃'))
        store.append(stable)
        context=ContextBuilder().build(store,stable)
        try:
            patch=translate_output(json.dumps(dict(base_revision=1,operations=[dict(op='replace',old_text='嗯啊啊呃',new_text='嗯，啊。')])),context.snapshot,'anchored')
            with self.assertRaisesRegex(PatchRejected,'ORAL_FILLER_REMOVED'):
                validate_segment_boundary(patch,context)
            self.assertEqual(store.canonical_text,'嗯啊啊呃')
        finally:store.close()

    def test_hard_cut_ending_is_rejected_but_internal_punctuation_is_allowed(self):
        store=TranscriptStore()
        stable=StabilityBuffer().push(ASREvidence('a',0,1,'如果可行我们就',end_reason='max_duration'))
        store.append(stable)
        context=ContextBuilder().build(store,stable)
        try:
            for text,should_reject in [('如果可行，我们就',False),('如果可行我们就。',True)]:
                patch=translate_output(json.dumps(dict(base_revision=1,operations=[dict(op='replace',old_text=context.snapshot.text,new_text=text)])),context.snapshot,'anchored')
                if should_reject:
                    with self.assertRaisesRegex(PatchRejected,'HARD_CUT_TERMINATOR'):
                        validate_segment_boundary(patch,context)
                else:validate_segment_boundary(patch,context)
        finally:store.close()


class FileImportTests(unittest.TestCase):
    def test_reader_backpressure_preserves_all_segments_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.wav'
            audio=np.full((32*16000,2),.1,np.float32)
            sf.write(source,audio,16000)
            original=source.read_bytes()
            capture=FileCapture(source,str(Path(folder)/'archive.wav'),threading.Event(),.008,
                                lambda _:None,lambda _:None,queue_size=1)
            capture.start()
            deadline=time.monotonic()+3
            while capture.segments.empty() and time.monotonic()<deadline: time.sleep(.01)
            self.assertFalse(capture.done.is_set())
            segments=[]
            while not capture.done.is_set() or not capture.segments.empty():
                try:segments.append(capture.segments.get(timeout=.1))
                except __import__('queue').Empty:pass
            capture.join(2)
            self.assertIsNone(capture.error)
            self.assertEqual(capture.dropped,0)
            self.assertEqual(sum(s.end_sample-s.start_sample for s in segments),len(audio))
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(sf.info(capture.wav_path).frames,len(audio))
            self.assertEqual(segments[-1].end_reason,'end_of_file')

    def test_failed_decoder_finishes_without_microphone(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'bad.wav';source.write_text('invalid')
            capture=FileCapture(source,str(Path(folder)/'out.wav'),threading.Event(),.008,lambda _:None,lambda _:None)
            capture.start();capture.join(2)
            self.assertTrue(capture.done.is_set())
            self.assertIsNotNone(capture.error)

    def test_abort_unblocks_full_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.wav';sf.write(source,np.full(32*16000,.1),16000)
            capture=FileCapture(source,str(Path(folder)/'out.wav'),threading.Event(),.008,lambda _:None,lambda _:None,queue_size=1)
            capture.start()
            time.sleep(.1)
            capture.abort.set();capture.join(2)
            self.assertFalse(capture.is_alive())


if __name__=='__main__': unittest.main()
