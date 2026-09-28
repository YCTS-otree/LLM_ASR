"""Instance-local adapter for FunASR 1.2.7; never edits site-packages.

Observe decoder hypotheses before upstream discards scores. Text is reconstructed
using the same token cleanup and non-timestamp postprocessor as the baseline.
"""
import math
import logging
from evidence import ASREvidence, Hypothesis

log = logging.getLogger('meeting_asr')


def normalize_timestamps(raw, duration):
    if raw is None:
        return (), 'timestamp_missing'
    try:
        pairs = []
        last = 0.0
        for pair in raw:
            if len(pair) != 2 or any(type(v) not in (int, float) for v in pair):
                raise ValueError()
            a, b = pair[0] / 1000, pair[1] / 1000
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= duration + .15 and a >= last):
                raise ValueError()
            pairs.append((a, b))
            last = a
        return tuple(pairs), None
    except (TypeError, ValueError, OverflowError):
        return (), 'timestamp_invalid'


def make_evidence(segment, results, hypotheses, latency, config, warning=None):
    results = results if isinstance(results, list) else []
    top = results[0] if results and isinstance(results[0], dict) else {}
    # Never concatenate alternate hypotheses into the canonical transcript.
    best = hypotheses[0].text if hypotheses else str(top.get('text', '')).strip()
    timestamps, problem = normalize_timestamps(top.get('timestamp'), segment.duration)
    return ASREvidence(segment.segment_id, segment.start_time, segment.end_time,
                       best, tuple(hypotheses), timestamps, results, segment.end_reason,
                       start_sample=segment.start_sample, end_sample=segment.end_sample,
                       sample_rate=segment.sample_rate, inference_latency=latency,
                       beam_size=config.beam_size, requested_nbest=config.nbest,
                       timestamp_warning=warning or problem, pause_after_ms=segment.pause_after_ms)


class ParaformerAdapter:
    def __init__(self, auto_model, config):
        from funasr.utils import postprocess_utils
        self.auto_model, self.config = auto_model, config
        self.model = model = auto_model.model
        self.hypotheses = []
        self.postprocess = postprocess_utils.sentence_postprocess
        self.tokenizer = auto_model.kwargs['tokenizer']
        kwargs = dict(auto_model.kwargs)
        kwargs.update(decoding_ctc_weight=0.0, lm_weight=0.0, beam_size=config.beam_size)
        model.init_beam_search(**kwargs)
        model.nbest = config.nbest
        self.original_search = model.beam_search.forward
        model.beam_search.forward = self._search

    def _search(self, *args, **kwargs):
        hyps = self.original_search(*args, **kwargs)
        for rank, hyp in enumerate(hyps[:self.config.nbest], 1):
            ids = hyp.yseq[1:-1].tolist() if hasattr(hyp.yseq, 'tolist') else hyp.yseq[1:-1]
            ids = [i for i in ids if i not in (self.model.sos, self.model.eos, self.model.blank_id)]
            tokens = self.tokenizer.ids2tokens(ids)
            text = (self.tokenizer.tokens2text(tokens) if hasattr(self.tokenizer, 'bpemodel')
                    else self.postprocess(tokens)[0])
            self.hypotheses.append(Hypothesis(rank, text.strip(), float(hyp.score)))
        return hyps

    def generate(self, audio):
        self.hypotheses = []
        try:
            result = self.auto_model.generate(input=audio, pred_timestamp=True, disable_pbar=True)
            return result, tuple(self.hypotheses), None
        except Exception:
            # Timestamp postprocessing can fail on token/alignment mismatches.
            # Retry text-only; if inference itself is broken the second error propagates.
            log.warning('timestamp-enabled inference failed; retrying text-only', exc_info=True)
            self.hypotheses = []
            result = self.auto_model.generate(input=audio, pred_timestamp=False, disable_pbar=True)
            return result, tuple(self.hypotheses), 'timestamp_fallback'

    def close(self):
        # Break the instance-level callback cycle before unloading a model.
        self.model.beam_search.forward = self.original_search
