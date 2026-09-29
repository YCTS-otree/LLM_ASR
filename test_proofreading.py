import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
import urllib.error
from evidence import ASREvidence, Hypothesis, StabilityBuffer
from transcript_store import TranscriptStore, TranscriptSnapshot
from llm_pipeline import CorrectionContext, MeetingPipeline
from qwen_backend import QwenLocalBackend
from settings import LocalLLMConfig
from llm_protocol import BackendResult
from focused_correction import focused_contexts
from llm_protocol import build_focused_messages, translate_focused_output
from patches import RevisionEngine, PatchRejected
from baseline_punctuation import BaselinePunctuation
from deepseek_backend import DeepSeekBackend, DeepSeekConfig, NoRedirect


class ProofreadingTests(unittest.TestCase):
    def test_pipeline_applies_later_context_to_old_sentence_without_changing_raw(self):
        store=TranscriptStore();backend=QwenLocalBackend(LocalLLMConfig());backend.ready=True
        def generate(c, deadline):
            combined=c.readable_context+c.snapshot.writable_text+c.readable_right
            target=c.snapshot.writable_text
            if '英文写作Beta' in combined:target=target.replace('贝塔','Beta')
            p=translate_focused_output(json.dumps(dict(base_revision=c.snapshot.revision,text=target)),c.snapshot,c.evidence)
            return BackendResult(p)
        backend._process_once=Mock(side_effect=generate)
        pipeline=MeetingPipeline(store,backend=backend)
        try:
            pipeline.accept(ASREvidence('a',0,1,'发布贝塔版本。'));pipeline.worker.requests.join()
            self.assertEqual(store.canonical_text,'发布贝塔版本。')
            pipeline.accept(ASREvidence('b',1,2,'这个名称英文写作Beta。'));pipeline.worker.requests.join()
            self.assertEqual(store.canonical_text,'发布Beta版本。这个名称英文写作Beta。')
            self.assertEqual(store.segments[0].raw_asr_text,'发布贝塔版本。')
        finally:pipeline.close();store.close()

    def test_context_can_correct_absent_candidate_and_normalize_numbers(self):
        for old,new in [('我们一九九九年成立','我们1999年成立'),
                        ('这款芯片是天机九千六','这款芯片是天玑9600'),
                        ('今天讨论锐龙九千系处理器','今天讨论锐龙9000系处理器'),
                        ('现在发布贝塔版本','现在发布Beta版本')]:
            with self.subTest(old=old):
                store=TranscriptStore()
                evidence=ASREvidence('s',0,1,old,nbest=(Hypothesis(1,old,None),))
                store.append(StabilityBuffer().push(evidence))
                s=store.snapshot()
                p=translate_focused_output(json.dumps(dict(base_revision=s.revision,text=new)),s,evidence)
                RevisionEngine(store).apply(p,s,'LOCAL_LLM')
                self.assertEqual(store.canonical_text,new)
                self.assertEqual(store.segments[0].raw_asr_text,old)
                store.close()

    def test_revisits_previous_completed_sentence_with_new_right_context(self):
        text='现在发布贝塔版本。'+'我们讨论这个产品的发布计划。'*12+'后文明确说英文名称是Beta。'
        context=CorrectionContext(TranscriptSnapshot(2,text,0,len(text)),ASREvidence('new',1,2,'后文明确说英文名称是Beta。'),segments=({'id':'new','start':len(text)-16},))
        parts=focused_contexts(context)
        self.assertEqual(parts[0].snapshot.window_start,0)
        self.assertIn('英文名称是Beta',parts[0].readable_right)
        self.assertEqual(''.join(p.snapshot.writable_text for p in parts),text)
        self.assertIn(parts[0].snapshot.writable_text,parts[1].readable_context)

    def test_long_candidate_not_filtered(self):
        text='完整识别候选'*35
        c=CorrectionContext(TranscriptSnapshot(1,'原文',0,2),ASREvidence('s',0,1,text,nbest=(Hypothesis(1,text,None),)),focused=True)
        self.assertIn(text,build_focused_messages(c)[-1]['content'])

    def test_whole_phrase_omission_is_rejected_but_duplicate_is_repairable(self):
        old='这一次这台手机有个很独特的地方就是它有三个传感器'
        s=TranscriptSnapshot(1,old,0,len(old));e=ASREvidence('s',0,1,old)
        with self.assertRaisesRegex(PatchRejected,'CONTENT_DRIFT'):
            translate_focused_output(json.dumps(dict(base_revision=1,text='就是它有三个传感器')),s,e)
        old='拍摄了几百百个视频';s=TranscriptSnapshot(1,old,0,len(old))
        self.assertTrue(translate_focused_output(json.dumps(dict(base_revision=1,text='拍摄了几百个视频')),s,e)['operations'])

    def test_baseline_punctuation_preserves_raw_and_survives_without_llm(self):
        store=TranscriptStore()
        e=ASREvidence('s',0,1,'你好世界',punctuated_text='你好，世界。')
        store.append(StabilityBuffer().push(e))
        self.assertEqual(store.canonical_text,'你好，世界。')
        self.assertEqual(store.segments[0].raw_asr_text,'你好世界')
        store.close()
        model=object.__new__(BaselinePunctuation);model.model=Mock()
        model.model.generate.return_value=[{'text':'你好，世界。'}]
        self.assertEqual(model.punctuate('你好世界','max_duration'),'你好，世界')
        model.model.generate.return_value=[{'text':' Oppo手机，很好。'}]
        self.assertEqual(model.punctuate('oppo手机很好','silence'),' oppo手机，很好。')
        model.model.generate.return_value=[{'text':'编造内容。'}]
        with self.assertRaises(ValueError):model.punctuate('你好世界','silence')


class DeepSeekTests(unittest.TestCase):
    def setup_backend(self):
        backend=DeepSeekBackend(DeepSeekConfig(api_key='unit-test-placeholder',thinking=True))
        backend.load()
        c=CorrectionContext(TranscriptSnapshot(1,'今天讨论芯片',0,6),ASREvidence('s',0,1,'今天讨论芯片'))
        return backend,c

    def test_request_uses_text_and_only_final_answer_is_applied(self):
        backend,c=self.setup_backend()
        response=Mock()
        response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        response.read1.side_effect=[json.dumps({'choices':[{'finish_reason':'stop','message':{
            'reasoning_content':'private reasoning must not be audited','content':'{"base_revision":1,"text":"今天讨论芯片。"}'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':20}}).encode(),b'']
        opener=Mock();opener.open.return_value=response
        with patch('deepseek_backend.urllib.request.build_opener',return_value=opener):
            result=backend.process(c)
        self.assertIsNone(result.error)
        self.assertTrue(result.patch['operations'])
        request=opener.open.call_args.args[0]
        data=json.loads(request.data)
        self.assertEqual(request.full_url,'https://api.deepseek.com/chat/completions')
        self.assertEqual(data['thinking'],{'type':'enabled'})
        self.assertNotIn('private reasoning',repr(result))
        self.assertNotIn('unit-test-placeholder',repr(result))
        self.assertNotIn('unit-test-placeholder',repr(backend.config))

    def test_http_error_is_redacted_and_not_retried(self):
        backend,c=self.setup_backend();opener=Mock()
        opener.open.side_effect=urllib.error.HTTPError('https://api.deepseek.com',401,'unit-test-placeholder',{},None)
        with patch('deepseek_backend.urllib.request.build_opener',return_value=opener):
            result=backend.process(c)
        self.assertEqual(opener.open.call_count,1)
        self.assertIsNone(result.patch)
        self.assertNotIn('unit-test-placeholder',repr(result))
        self.assertIn('HTTP_401',repr(result))
        self.assertIsNone(NoRedirect().redirect_request(None,None,302,'',{},'https://other.invalid'))

    def test_missing_key_does_not_call_network(self):
        with patch('deepseek_backend.urllib.request.build_opener') as network:
            with self.assertRaises(ValueError):DeepSeekBackend(DeepSeekConfig()).load()
            network.assert_not_called()
