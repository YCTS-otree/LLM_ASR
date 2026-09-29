# v1.6.0 verification — 2026-09-29

This is a functional verification, not meeting-quality acceptance or a live
DeepSeek latency benchmark. User reports that DeepSeek Flash corrects names and
contextual word errors much better; those observations were not independently
scored in this run.

## Automated and UI checks

-119 tests passed with the existing Python3.10 environment. Coverage includes
  effective API payload settings, invalid limits, staged GUI settings without
  automatic model loading, larger proofreading targets, pause provenance,
  terminology retrieval, numeric formatting and existing transcription behavior.
- API tests use mocks and temporary dummy credentials. No real key was inspected
  and no live API request was made for this verification.
- Main single/dual/API layouts were rendered at900×700 and720×600. DeepSeek request
  settings and glossary editor were visually checked for clipping and overlap.

## Local formatting probe

Six synthetic cases were run on the existing Qwen3.5-2B BF16 model. Prompt-only
behavior passed2/6; the model left 二纳米、八千毫安时、两亿像素、锐龙九千系
unchanged. These were valid no-change answers, not validator rejection.

With narrow written-number rules enabled, the same suite passed6/6: 2nm,
8000mAh,2亿像素,1999年,锐龙9000系, and unchanged 一心一意/三思而后行.
Four cases were rule-assisted. This does not establish improved semantic
reasoning or knowledge in the model. Ambiguous colloquial quantities and
unrelated idioms are preserved; semantic repairs such as 我妹→我们 remain the
LLM's contextual responsibility, never a global substitution rule.

## Real60-second import

The previously authorized60-second excerpt from the user's professional phone
review was processed through the Qt import flow with Paraformer FP16, independent
CPU punctuation, Qwen2B BF16, combined punctuation mode and a2048-character window.

- Four segments, zero ASR failures, zero baseline-punctuation failures, zero queue
  drops. Source checksum unchanged; UI, SQLite and TXT matched.
- Final text had27 punctuation marks. Count is not punctuation accuracy.
- CUDA allocated peak approximately5035.8MiB, not total process/GPU memory.
- At the test RMS threshold0.001, no120ms low-energy spans were detected. Three
  segments had usable raw word/timestamp alignment; one did not. Timestamp-gap
  fallback yielded two hints in one segment. No word anchors were fabricated for
  the unaligned segment.
- No fresh reference error-rate score was calculated. The bundled glossary
  includes the user's OPPO model example; any assisted name recovery would not
  be an independent blind recognition result. No audio, reference subtitles,
  databases or user glossary are published.

## API latency scope

Previous API configuration enabled thinking without an explicit effort, and
reused128-character targets. New DeepSeek defaults use512-character targets and
disable the optional second punctuation pass after a valid no-change answer.
Effort, output budget, request/batch timeout and target length are configurable;
metadata records request count and reasoning tokens. Mock transport confirms
settings reach requests, but service/network latency and quality tradeoffs still
require user testing. Defaults retain high effort rather than silently reducing
reasoning quality. Larger targets remain subject to unchanged patch safety limits.
