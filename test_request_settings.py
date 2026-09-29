import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
from deepseek_backend import DeepSeekBackend, DeepSeekConfig
from evidence import ASREvidence
from focused_correction import focused_contexts
from llm_pipeline import CorrectionContext
from llm_protocol import translate_focused_output
from transcript_store import TranscriptSnapshot
from number_formatting import normalize_written_numbers
from glossary import relevant_terms, validate_terms


class RequestSettingsTests(unittest.TestCase):
    def test_large_targets_keep_context_and_allow_small_edits(self):
        text='这是测试文字。'*160
        context=CorrectionContext(TranscriptSnapshot(2,text,0,len(text)),ASREvidence('s',0,1,text))
        small=focused_contexts(context,128);large=focused_contexts(context,512)
        self.assertLess(len(large),len(small))
        self.assertEqual(''.join(c.snapshot.writable_text for c in large),text)
        self.assertIn(large[0].snapshot.writable_text,large[1].readable_context)
        s=large[0].snapshot
        result=translate_focused_output(json.dumps(dict(base_revision=2,text=s.writable_text.replace('。','，',1))),s,context.evidence)
        self.assertEqual(len(result['operations']),1)
        self.assertEqual(result['operations'][0]['text'],'，')

    def test_api_parameters_are_effective_and_inapplicable_fields_omitted(self):
        for thinking in (True,False):
            config=DeepSeekConfig(api_key='unit-test-placeholder',thinking=thinking,reasoning_effort='low',
                output_token_limit=4096,request_timeout=45,target_chars=1024,temperature=.3)
            backend=DeepSeekBackend(config);backend.load()
            c=CorrectionContext(TranscriptSnapshot(1,'你好',0,2),ASREvidence('s',0,1,'你好'))
            response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
            response.read1.side_effect=[json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"base_revision":1,"text":"你好。"}'}}],
                'usage':{'completion_tokens_details':{'reasoning_tokens':12}}}).encode(),b'']
            opener=Mock();opener.open.return_value=response
            with patch('deepseek_backend.urllib.request.build_opener',return_value=opener):result=backend.process(c)
            payload=json.loads(opener.open.call_args.args[0].data)
            self.assertEqual(payload['max_tokens'],4096)
            self.assertLessEqual(opener.open.call_args.kwargs['timeout'],45)
            self.assertEqual(result.metadata['reasoning_tokens'],12)
            if thinking:
                self.assertEqual(payload['reasoning_effort'],'low');self.assertNotIn('temperature',payload)
            else:
                self.assertEqual(payload['temperature'],.3);self.assertNotIn('reasoning_effort',payload)

    def test_invalid_request_limits_fail_before_network(self):
        for kwargs in ({'target_chars':0},{'target_chars':2049},{'reasoning_effort':'medium'},
                       {'output_token_limit':99999},{'request_timeout':0},{'temperature':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):DeepSeekConfig(**kwargs)

    def test_number_rules_preserve_values_and_ambiguous_language(self):
        for old,new in [('二纳米','2nm'),('八千毫安时','8000mAh'),('两亿像素','2亿像素'),
                        ('一九九九年','1999年'),('锐龙九千系','锐龙9000系'),('三点五吉赫兹','3.5GHz'),
                        ('三千五毫安时','三千五毫安时'),('一心一意三思而后行我妹来了','一心一意三思而后行我妹来了'),
                        ('二纳米比亚','二纳米比亚')]:
            self.assertEqual(normalize_written_numbers(old),new)

    def test_terminology_retrieval_is_contextual_and_not_replacement(self):
        terms=validate_terms([dict(term='OPPO Find X10 Pro Max',aliases=['oppe find x十pro max'],note='参考')])
        self.assertEqual(relevant_terms('oppe find x十pro max',terms),terms)
        self.assertFalse(relevant_terms('我们今天讨论会议安排',terms))
        self.assertEqual(normalize_written_numbers('oppe find x十pro max'),'oppe find x十pro max')
