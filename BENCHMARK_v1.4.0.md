# v1.4.0 — long-input punctuation regression

Date: 2026-09-29. Windows / Python 3.10.11 / RTX 4060 Laptop 8GB,
Torch 2.5.1+cu121, Transformers 5.17.0. Existing Paraformer and local
Qwen3.5-2B BF16 weights; no new runtime or model family.

## Original failure

A recent 38-segment offline session had one LLM request per ASR segment.
26 requests reached the 128-token output cap, three failed anchor validation,
and two failed the hard-cut guard. Only seven requests applied a patch.
The offline pipeline already waited for correction after each segment.
The dominant failure was rejected model output, not ASR overtaking the LLM.

## Same-evidence replay

The immutable ASR records from the same session were replayed in a new database.
No ASR was rerun in this comparison. The new configuration uses a 2048-character
writable window, up to 64-character targets, a 192-token per-attempt cap, a
90-second per-batch deadline and one punctuation-only retry where needed.

| Observation | Original v1.3.0 | v1.4.0 replay |
|---|---:|---:|
| ASR segments / top-level correction requests | 38 / 38 | 38 / 38 |
| Segments containing punctuation | 7 | 28 |
| Punctuation characters | 27 | 104 |
| Applied / no-change top-level requests | 7 / 0 | 32 / 6 |
| Model generation calls, including retries | 38 | 164 |
| Non-punctuation, non-whitespace characters | 3244 | 3244 |
| Segments with lexical text different from raw ASR | 0 | 0 |

The new run had 103 targets, with 11 targets still rejected for content drift
after retry; other targets in those batches could apply. Across all attempts,
43 content-drift failures, four output-token-limit failures, four hard-cut
failures and one invalid-output failure were recorded. The latter three kinds
were recovered or avoided on retry. Mean batch latency was 7.745 seconds on
this shared GPU. Retries and guards improve coverage but increase work.

Punctuation counts are coverage signals, **not accuracy scores**. No manually
annotated reference or punctuation precision/recall was measured. Inspection
still found missing punctuation and imperfect sentence boundaries. ASR lexical
errors and repetitions were preserved. This version does not claim complete
punctuation recovery or general meeting-transcription accuracy.

An earlier compact-output experiment copied context and repeated words despite
many APPLIED results. It was not accepted for release. The published guard and
final replay check lexical fidelity instead of treating APPLIED as quality.

## Actual file import

A local 582.741-second stereo MP3 was read without modification; only its first
60 seconds were copied to a local 48kHz PCM16 WAV for an end-to-end test. The
recording, transcript and original filename are intentionally not published.

Qt import -> Paraformer FP16 -> Qwen3.5-2B BF16 -> SQLite/TXT was exercised with
a 2048-character window. It produced three ASR segments and eight punctuation
characters, with zero queue drops. The test prohibited microphone opening and
verified unchanged input, matching UI/SQLite/TXT text, one model load, and correct
file mode. Own CUDA peak allocation was approximately 4745 MiB; total device
memory use can be larger. This is a functional test, not a full transcription
quality acceptance test. Model comparison and NF4 quality were not rebenchmarked
in this iteration; older measurements remain version-specific.

## Verification and reproducibility

- Unit/integration coverage includes immutable raw evidence, frozen boundaries,
  strict revisions/offsets, content-drift rejection, cancellation, partial target
  failure, persisted 2048-character window and offline ASR/LLM ordering.
- Actual Qt single/dual layouts inspected at 900x700 and 720x600, including
  2048-character control and partial-failure status.
- Use `python replay_evidence.py path/to/session.sqlite3 --window-chars 2048`
  for an independent local result; use `verify_file_import.py --path clip.wav
  --dtype bf16 --mode 2b --window-chars 2048` for the full pipeline.
- User audio, session databases and detailed model logs remain local and are
  excluded from source publication. Only aggregate observations appear here.
