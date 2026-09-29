# v1.5.0 — contextual proofreading and optional DeepSeek

Date: 2026-09-29. Same local Paraformer weights and Qwen3.5-2B BF16;
CPU CT-Transformer punctuation added from ModelScope. Only a previously
authorized 60-second professional voiceover excerpt was used. The separately
edited ASS reference was used for evaluation, never given to either model.

## Final local end-to-end check

- Four segments cover [0,15], [15,30], [30,45], [45,60].
- Zero ASR failures, punctuation failures or queue drops; one FP32 recovery.
- Source unchanged; UI, SQLite and TXT match; GUI at900x700 and720x600.
- Own CUDA allocation peak4746.45 MiB (not total system VRAM).
- Baseline punctuation23 marks; final25 marks. This counts marks, not accuracy.
- All four correction requests applied patches; final-target failures0.
- Request latencies3.72,11.26,15.23,24.03 seconds, increasingly revisiting history.

| Text | Reference characters | Edit distance | Difference rate |
|---|---:|---:|---:|
| Raw ASR |379|37|9.76%|
| Independent punctuation |379|37|9.76%|
| Qwen corrected |379|37|9.76%|

The metric removes punctuation/whitespace and folds Unicode/case. It does not
normalize spoken numerals into digits. The reference is user-edited AI output,
not an independently verified verbatim gold transcript. **Qwen brought no lexical
gain in this test.** Punctuation placement also remains imperfect. This is not
meeting-quality acceptance and does not establish that contextual correction is
impossible with other models or prompts.

An earlier relaxed-deletion candidate omitted a phrase and increased difference
to14.25%; it was rejected. Final guards reject non-duplicate pure lexical
deletion and large replacements. A punctuation-only model capitalized an English
initial letter; final punctuation transfer preserves original lexical characters
so this no longer discards an otherwise useful paragraph of punctuation.

## Thinking and API limits

A Qwen thinking probe exhausted4096 output tokens without completing its final
answer on the first target (batch253.90 seconds, including retry). It was stopped
before completing the four-segment replay. No complete thinking-mode accuracy
claim is made. Thinking remains optional/off by default; punctuation recovery
uses non-thinking mode. See the [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-2B)
for its warning about possible thinking loops in2B.

DeepSeek request construction, final-answer handling, redacted HTTP failure,
no retry after authentication failure, and redirect rejection were tested with
mocked responses. **No real API call or online quality benchmark was performed:
no user API key was provided.** The settings dialog supports a model ID and
thinking switch using the [official chat API](https://api-docs.deepseek.com/api/create-chat-completion/).

## Regression scope

Tests cover contextual corrections absent from candidates, numeral conversion,
later-context edits to an already stored sentence without changing raw ASR,
overlapping read context/nonoverlapping writes, preservation of full candidates,
baseline punctuation independent of LLM, stale/frozen/overlap checks, explicit
loading, API staging and secret-free metadata. These simulated semantic outputs
test the software path, **not the language model's ability** to produce them.

Audio still uses15-second maximum segments; no acoustic overlap/second ASR pass
was added. Read-context overlap fixes the LLM's narrow view, not missing audio.
No fresh0.8B/dual/NF4 quality benchmark in this iteration. Audio, references,
transcripts, raw output and model weights remain excluded from publication.
