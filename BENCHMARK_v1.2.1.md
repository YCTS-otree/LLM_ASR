# v1.2.1 punctuation and offline import

Date: 2026-09-28. Existing RTX 4060 Laptop 8GB, Python 3.10.11,
Torch 2.5.1+cu121, FunASR 1.2.7, Transformers 5.17.0. Same official
Qwen3.5-2B text weights; BF16 default and optional NF4. No new runtime or model.

## Prompt comparison

Twelve hand-authored development cases: P1–P10, with P5 split into unfinished
and continued cases, plus spoken prompt injection. Eleven require punctuation;
P5a must preserve an unfinished hard cut. These are development examples used
while tuning, not an independent/general accuracy benchmark.

| Mode / prompt | Punctuation cases | System behavior | Changed wording | Invalid | No change | Guard rejection | Mean latency | Output tokens | Approx J/request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 / v5 | 0/11 | 1/12 | 0/12 | 0 | 12 | 0 | 0.50s | 12.00 | 17.83 |
| BF16 / v6 | 11/11 | 12/12 | 0/12 | 0 | 1 | 0 | 1.31s | 35.83 | 55.71 |
| NF4 / v5 | 0/11 | 1/12 | 0/12 | 1 | 11 | 0 | 0.41s | 12.00 | 14.87 |
| NF4 / v6 | 11/11 | 12/12 | 0/12 | 0 | 0 | 1 | 1.08s | 38.17 | 45.24 |

NF4's P5a model output incorrectly adds a period; HARD_CUT_TERMINATOR rejects
that entire request. Thus model output itself passed 11/12, while preserved
application behavior passed 12/12. BF16 passed all 12 without that guard firing.
Wrong-word and polishing proxies are exact wording comparisons after removing
basic punctuation; both are 0/12 on this set only. Original spaces are preserved.

Mean input length rose from 436.92 to 661.17 tokens. Output rose about 3x because
v5 mostly returned an empty patch. Latency and measured energy also rose about
2.6–3.1x: this is a material cost, not a negligible increase. Local short-span
replacements improve reliability but repeat text on the wire; deterministic diff
reduces stored edits, not already-generated tokens. The 128-token cap is unchanged.

GPU energy is whole-device sampled power integrated approximately over request
time, not incremental model energy. Another user Python GPU process was present;
Windows desktop/background activity and sparse samples limit precision. Tests
were sequential, single-run, with v5 then v6 on the same resident model. Do not
infer a universal NF4 energy advantage. Raw local records are under logs/;
shareable aggregates are in benchmarks/v1.2.1_summary.json.

The v5 formatter/system text are copied from tag v1.2.0. Both prompt versions use
the current translator/validator, isolating prompt behavior rather than comparing
two complete historical applications. Final filler protection was added after
these measurements; none of these outputs removes fillers, so outcomes stand.

## Original lexical cases and live behavior

INT4 rerun of A–H: 5/8 passed. A misses the supported H170 correction and changes
sentence punctuation; D still misses later-context correction; G reaches the
output-token cap and is safely rejected. Frozen content is intact. This is a
regression from the v1.2.0 INT4 7/8 development result, and general lexical quality
is not accepted merely because the punctuation examples pass.

A real 30.2-second RODE microphone run at threshold 0.001 captured 3 nonempty
ASR segments, 3 Qwen calls and punctuation, with no dropped segments or device
overflow. Speech level was low and the captured conversation did not cover the
requested question/long-clause scenarios. One model patch removed a repeated
spoken filler. Added ORAL_FILLER_REMOVED rejects reductions in 嗯/啊/呃 counts,
preserving the entire original request on rejection. This conservative rule can
also reject a legitimate duplicate removal; it does not guarantee general semantic
fidelity. Live speech correctness and all required sentence types remain unproven.
Personal speech/audio remain local and are excluded from Git.

Offline replay of that same recording after the guard passed: three identical raw
ASR segments, two applied punctuation patches, one ORAL_FILLER_REMOVED rejection,
no lost fillers, no dropped segments, matching UI/TXT and unchanged source hash.

## Offline import and infrastructure

Real Qt import of the official Paraformer WAV passed: one ASR segment, one applied
Qwen patch, final “欢迎大家来体验达摩院推出的语音识别模型。”, no microphone stream,
no dropped segments, matching UI/SQLite/TXT, unchanged source SHA256. An EOF enum
compatibility issue found by this end-to-end test was fixed and retested.

All 74 automatic tests passed, covering source preservation, bounded-queue backpressure, decode
failure, producer abort, EOF, minimal punctuation edits, segment metadata,
unchanged size/frozen/overlap guards, hard-cut rejection and oral filler retention.
Qt layouts are inspected at 900×700 and 720×600. No full release build is produced;
source commits/tags are the rollback points. Long meetings and broad audio-format
coverage still require further testing.
