# v1.3.0: Qwen3.5-2B / 0.8B comparison

Date: 2026-09-28. Same existing RTX 4060 Laptop 8GB / Torch / Transformers
environment as v1.2.1. No additional runtime packages. Official 0.8B downloaded
from ModelScope (1,746,942,600-byte safetensors), verified against official HF
revision `2fc06364715b967f1860aea9cf38778875588b17` and SHA256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
Both use the text-only Qwen3_5ForCausalLM class, with no vision tower.

## Same-prompt development cases

Same 12 cases and v6 prompt as v1.2.1; no small-model-specific prompt tuning.
Same anchored protocol, greedy generation, no thinking and 128-token output cap.
These are small development examples, not a general transcription accuracy score.

| Model / precision | Behavior passed | Punctuation required passed | Invalid output | No change | Mean latency | Mean output tokens | Approx J/request |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2B BF16 | 12/12 | 11/11 | 0 | 1 | 1.31s | 35.83 | 55.71 |
| 2B NF4 | 12/12* | 11/11 | 0 | 0 | 1.08s | 38.17 | 45.24 |
| 0.8B BF16 | 5/12 | 4/11 | 1 | 3 | 1.09s | 36.50 | 38.45 |
| 0.8B NF4 | 2/12 | 1/11 | 3 | 2 | 1.74s | 46.08 | 46.90 |

*2B NF4 has one hard-cut terminator rejected by the guard; raw model output was
11/12. The other application behavior remains correct because original text is
retained. 2B figures are the unchanged v1.2.1 measurements, not reruns in the same
thermal state. Power is sampled whole-device approximate energy with another user
GPU process present, so these figures are instrumentation, not isolated energy
efficiency claims. See BENCHMARK_v1.2.1.md for methodology and lexical regressions.

0.8B BF16 often omits internal commas, uses a period for a question, or misses
punctuation entirely. Its one strict wording mismatch adds spaces around Whisper;
it is a spacing-policy violation, not a spoken-word substitution. NF4 frequently
uses a trailing comma and has three malformed/token-limited outputs. The smaller
model is therefore offered for comparison, not recommended as an equivalent
quality replacement. Quantization did not improve latency on this example set.

## Real audio and concurrent UI verification

Imported the official Paraformer WAV through the actual Qt button. The microphone
API was mocked to raise if opened. ASR ran once per input segment; both stores
contain identical complete raw evidence and segment IDs, independent revisions,
and their own request metadata. Source hash was unchanged, no segments dropped,
and each panel exactly matched its SQLite canonical text and exported TXT.

- Both NF4: 2B applied punctuation; 0.8B hit the 128-token cap and retained ASR.
  Request intervals overlapped about 3.10s; peak process CUDA allocation 3528 MiB.
- 2B NF4 + 0.8B BF16: both applied the expected sentence-ending period. Two
  consecutive imports completed with each model load_count still 1. The last
  session's requests took 5.18s and 5.00s and overlapped 5.00s. Peak process CUDA
  allocation was 4241 MiB (4.14 GiB). This became the default comparison precision.
- 0.8B BF16 alone also produced the correct punctuation on the official audio.

These are overlapping independent requests on one shared GPU, not proof of
simultaneous GPU kernels or speedup. Comparison was slower than individual model
tests; competing GPU memory/work and CPU dispatch affect latency. Peak allocated
memory excludes other processes and CUDA driver/reserved allocations. Short sample
success does not establish long-meeting memory headroom or sustained throughput.

The two pipelines evolve their own recent corrected contexts. They share ASR,
not each other's edits; comparisons after the first request therefore assess
complete pipeline behavior. Offline input waits for both workers for every segment.
Live recording remains asynchronous and may coalesce differing request counts.

## Verification and rollback

All 79 automated tests passed, exercising genuine two-worker concurrency with a barrier, one ASR
call feeding two distinct corrections, independent exports, one-sided LLM failure,
load failure isolation, group cancellation, model/precision selection and existing
frozen/revision safety checks. The real dual layout was inspected at 900×700 and
720×600, including rejected-output and successful-output states.

The original v1.2.1 checkpoint remains at tag v1.2.1; development occurred on
codex/dual-qwen-comparison. No executable release build or remote push is included.
Raw personal audio and session records remain ignored locally. Aggregate model
metrics are in benchmarks/v1.3.0_summary.json.
