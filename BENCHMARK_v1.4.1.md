# v1.4.1 — completeness and independent reference check

Date: 2026-09-29. Same local Paraformer weights and Qwen3.5-2B BF16.
The user supplied a professional voiceover and separately edited AI subtitles.
Only the first 60 seconds were tested end-to-end. The ASS reference was used
after transcription for evaluation, never supplied to the ASR or LLM.

## Missing opening segment

The previous 60-second test saved three segments beginning at 15 seconds.
SQLite contained an ASR_FAILED event for 0–15 seconds: CIF raised an IndexError
under FP16. The same audio was independently tested with the existing model:
FP16 failed; BF16 and FP32 both transcribed the opening. This establishes
precision-dependent behavior for this sample, not its general frequency.

v1.4.1 retries an AMP IndexError once on the same samples and resident weights
with autocast disabled. Successful recovery records actual precision and reason
in immutable evidence and an ASR_PRECISION_FALLBACK event. Persistent failure
still records ASR_FAILED. There is no model substitution or fabricated text.

The corrected real Qt import produced four segments covering [0,15], [15,30],
[30,45], [45,60], with **zero ASR failures**, zero queue drops and matching
UI/SQLite/TXT. One segment used FP32 recovery. The source excerpt was unchanged.
The verifier now fails on any ASR_FAILED event. The former test's passed flag
was insufficient and is explicitly corrected in BENCHMARK_v1.4.0.md.

## Reference comparison

Comparison uses NFKC, case folding, and removal of punctuation and whitespace.
Levenshtein distance divided by reference length gives the character difference
rate. Chinese/Arabic numerals and product-name spelling are not normalized, so
those formatting differences also count. The 379-character reference is edited
AI output, not an independently verified verbatim gold transcript.

| Output, first 60 seconds | Edit distance | Difference rate |
|---|---:|---:|
| v1.4.0 raw ASR / corrected | 143 / 143 | 37.73% / 37.73% |
| v1.4.1 raw ASR / corrected | 37 / 37 | 9.76% / 9.76% |

The improvement comes from recovering missing speech. **The current LLM did not
improve lexical agreement on this sample.** Remaining homophone errors,
repetitions and punctuation problems prevent claiming meeting-ready quality.
Punctuation count and a validated patch are not correctness measurements.

A local, evaluation-only experiment allowing bounded contextual substitutions
also left edit distance at 37 and reduced punctuation coverage. It was not
adopted. No reference-derived glossary or corrections were injected to inflate
the result. Broader held-out audio and a verified transcript are still needed.

`compare_subtitles.py reference.ass session.sqlite3 --end 60 --output
logs/comparison.json` reproduces the comparison on local files. Its optional
output contains reference and transcript text: keep it private. No user audio,
ASS subtitle or transcript is included in this repository.

## Manual model loading

Startup and parameter changes do not load models. Select all parameters, then
click Load model. The button loads/reuses ASR first, then enabled Qwen models.
Model selection and precision edits remain pending until this action; they do
not unload current weights or relabel previous results. Starting with unloaded
or changed model settings prompts for explicit loading. Re-enabling correction
does not trigger loading. Threshold/output/window changes do not reload weights.

Automated regression and actual Qt layout checks cover the button, staged
configuration, blocked premature capture, precision recovery and reference
comparison. Both 900x700 and 720x600 single/dual layouts were inspected.
