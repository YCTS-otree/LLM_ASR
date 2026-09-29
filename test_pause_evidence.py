import json
import unittest
from dataclasses import replace
import numpy as np
from evidence import ASREvidence,StabilityBuffer
from pause_evidence import energy_pauses,pause_hints
from llm_pipeline import ContextBuilder,MeetingPipeline
from transcript_store import TranscriptStore
from focused_correction import focused_contexts
from llm_protocol import build_focused_messages


class PauseEvidenceTests(unittest.TestCase):
    def evidence(self):
        return ASREvidence('s',10,12,'今天开会',timestamps=((0,.2),(.2,.4),(.8,1),(1,1.2)),
            raw_result=[{'text':'今 天 开 会'}],low_energy_spans=((.4,.8),),pause_threshold=.008,
            pause_after_ms=800,punctuated_text='今天，开会。')

    def test_energy_spans_exclude_short_dips_and_measure_trailing_pause(self):
        audio=np.concatenate([np.full(400,.1),np.zeros(200),np.full(300,.1),np.zeros(100),np.full(200,.1),np.zeros(200)]).astype(np.float32)
        self.assertEqual(energy_pauses(audio,1000,.008),((.4,.6),(1.2,1.4)))

    def test_raw_anchor_and_global_timing_are_explicitly_approximate(self):
        hints=pause_hints(self.evidence().to_dict())
        p=hints['internal_pauses'][0]
        self.assertEqual((p['before'],p['after']),('今天','开会'))
        self.assertEqual((p['start_s'],p['end_s'],p['duration_ms']),(10.4,10.8,400))
        self.assertEqual(p['alignment'],'approximate_raw_asr_anchor')
        self.assertEqual(hints['trailing_low_energy_ms'],800)

    def test_bad_token_alignment_never_fabricates_word_positions(self):
        e=replace(self.evidence(),raw_result=[{'text':'错误文本'}])
        h=pause_hints(e.to_dict())
        self.assertFalse(h['word_alignment_available'])
        self.assertEqual(h['internal_pauses'][0]['alignment'],'time_only')
        self.assertNotIn('before',h['internal_pauses'][0])

    def test_historical_timestamps_can_supply_labelled_gap_without_audio(self):
        h=pause_hints(replace(self.evidence(),low_energy_spans=()).to_dict())
        self.assertEqual(h['internal_pauses'][0]['kind'],'asr_timestamp_gap')

    def test_three_modes_select_initial_text_and_prompt_evidence(self):
        class Disabled:
            enabled=False
        for mode in ('baseline','pauses','combined'):
            store=TranscriptStore();pipeline=MeetingPipeline(store,backend=Disabled(),punctuation_mode=mode)
            try:
                stable=pipeline.accept(self.evidence())
                self.assertEqual(store.canonical_text,'今天开会' if mode=='pauses' else '今天，开会。')
                self.assertEqual(store.evidence('s')['punctuated_text'],'今天，开会。')
                c=focused_contexts(ContextBuilder().build(store,stable,mode))[0]
                for punctuation_only in (False,True):
                    messages=build_focused_messages(replace(c,punctuation_only=punctuation_only))
                    payload=json.JSONDecoder().raw_decode(messages[-1]['content'])[0]
                    self.assertEqual(payload['punctuation_strategy'],mode)
                    self.assertEqual(bool(payload['pause_evidence']),mode!='baseline')
                    if mode!='baseline':self.assertIn('400',json.dumps(payload['pause_evidence']))
            finally:pipeline.close();store.close()

    def test_old_raw_anchors_survive_edited_canonical_text(self):
        e=self.evidence().to_dict();e['best_text']='今天开会'
        h=pause_hints(e)
        self.assertEqual(h['raw_asr'],'今天开会')
        self.assertEqual(h['internal_pauses'][0]['before'],'今天')
