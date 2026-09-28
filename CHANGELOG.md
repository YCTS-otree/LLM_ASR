## v1.0.0 - 2026-09-28

### Added
- Existing functional FunASR transcription application baseline.
- Explicit application version and exclusions for generated output and local secrets.

## v1.1.0 - 2026-09-28

### Added
- Typed AudioSegment/ASREvidence, native sample positions, explicit segment end reasons, ordered stable evidence and configurable N-best with original decoder scores.
- SQLite TranscriptStore with immutable raw ASR/evidence, canonical character mapping, durable revision/patch history and TXT export.
- Monotonic 1024-Python-character writable window, strict replace/insert/delete protocol, atomic validation/application, size guards and stale-response rejection.
- ContextBuilder, shared LLMBackend interface and bounded asynchronous MockLLM correction worker; no real LLM runtime or online service added.
- Rotating operational logs, ASR duration/latency/beam instrumentation, queue-overflow and failed-segment recovery coordinates.
- Architecture documentation, automated regression coverage and microphone/UI verification helpers.

### Changed
- FunASR loads in the background at application startup and remains resident across recording sessions; explicit device/precision changes reload it.
- Existing UI reads the latest canonical transcript, reflects patches and identifies Mock correction mode.
- Audio queue overflow keeps recording while marking skipped ASR segments; audio uses RF64 WAV for long-file capacity.
- Individual inference failures are audited without terminating subsequent segment processing; window close flushes and saves before exiting.

### Fixed
- Alternate ASR hypotheses no longer concatenate into the transcript.
- Timestamp-format differences and missing/invalid alignment no longer corrupt the best-text path; timestamp failures retry text-only.
- Frozen text cannot re-enter the writable window after deletion; late correction responses cannot overwrite newer ASR revisions.

### Notes
- All 47 automatic tests passed, including cross-segment edits, transaction rollback, Unicode boundaries, stale responses, model reuse and UI refresh.
- Official sample passed on CPU FP32/INT8 and CUDA FP32/FP16/BF16; separate beam 1/3/5 checks passed through the real ASR-to-Store path. Latencies are instrumentation only, not a controlled benchmark.
- Two real microphone recordings (about 5.2 seconds each) passed capture, level signals, Start/Stop, local file output and same-model reuse, without overflow. No speech crossed the segmentation threshold, so these do not establish live speech accuracy.
- Patched transcript UI rendered and inspected at 900x700 and 720x600. Multi-hour power/stability tests, automatic backlog recovery and Undo UI remain future work.

## v1.2.0 - 2026-09-28

### Added
- Experimental resident Qwen/Qwen3.5-2B text-only correction through Transformers CUDA BF16, with optional bitsandbytes NF4 and explicit local model configuration.
- Opt-in official ModelScope/Hugging Face download with verified safetensors SHA256; application loads local files only and never downloads on startup.
- Strict relative-offset and unique-anchor JSON translators feeding the unchanged canonical PatchValidator, with versioned external prompt and request metadata in audit history.
- GPU power/VRAM sampling, eight model-specific development cases, ten-minute Qt/ASR/LLM/SQLite soak, live microphone verification and a shareable aggregate benchmark report.
- Local correction enable/status controls, log entry, and a read-only session evidence/request/patch inspection utility.

### Changed
- Application sessions use the real Qwen backend; Mock remains available for deterministic infrastructure tests.
- Correction requests use a bounded latest-request queue with a fixed 0.35-second debounce and separate readable/writable context.
- Default model-facing protocol uses exact unique anchors after measured relative-offset failures; default precision remains BF16, with INT4 configurable independently of ASR precision.

### Fixed
- Optional LLM configuration/load/runtime/OOM/timeout/output failures preserve ASR operation and original transcript; disabled correction suppresses new work without unloading resident weights.

### Notes
- All 63 automatic tests and dependency consistency checks passed. Existing Torch 2.5.1+cu121/FunASR 1.2.7 were preserved; five ASR runtime combinations were verified after adding Transformers.
- The 600-second soak processed 60 segments with one Qwen load, zero stale responses and healthy SQLite/UI; three invalid outputs were rejected. Real 30.2-second microphone capture produced two speech segments and two INT4 requests, without dropped audio or device overflow.
- Quality is not fully accepted: BF16 passed 6/8 development cases and NF4 7/8; later-context correction still failed. Live speech returned one no-op and one invalid output. This is an integration/measurement baseline, not a claim of production correction accuracy.
- Benchmark, dependency additions, protocol tradeoffs and known limitations are documented in BENCHMARK_v1.2.0.md. No online mode or ASR model switch was introduced.

## v1.2.1 - 2026-09-28

### Added
- Local recording-file import through the existing ASR/Qwen pipeline, with backpressure, read-only source handling and automatic SQLite/TXT export at EOF.
- Segment-boundary and low-cost trailing-pause evidence for punctuation; guards against forced hard-cut terminators and removal of spoken 嗯/啊/呃.
- Punctuation comparison benchmark, offline import verification and segment-preserving minimal-edit tests.

### Changed
- Qwen prompt v6 actively restores punctuation while retaining conservative lexical rules and oral wording.
- Validated local replacements are deterministically minimized and revalidated; boundary punctuation belongs to the preceding segment.

### Fixed
- Missing punctuation on correct ASR text; redundant no-op replacements no longer create revisions.
- EOF segment validation and producer shutdown for file input.

### Notes
- All 74 automatic tests passed; real Qt file import and 30.2-second microphone capture verified, followed by successful filler-guard replay. Two UI sizes inspected.
- BF16 and NF4 restored punctuation in 11/11 development examples; NF4 needed one additional hard-cut rejection. Original INT4 lexical cases regressed to 5/8. Live sentence-type coverage and broad semantic quality remain unaccepted.
- Mean output tokens and approximate request energy grew about threefold versus the mostly no-op v5 prompt. Measurements and limitations are in BENCHMARK_v1.2.1.md.

## v1.3.0 - 2026-09-28

### Added
- Official local Qwen3.5-0.8B selection and explicit ModelScope download with pinned SHA256 verification; existing runtime dependencies reused.
- Concurrent 2B/0.8B comparison with shared ASR evidence, independent transcript stores, revision histories and TXT/SQLite exports.
- Equal-width result panes with request latency/outcome and independently selectable model precision. Default comparison uses 2B NF4 plus 0.8B BF16.
- Real dual-model import/reuse checks and model-quality comparison report.

### Changed
- Model lifecycle now supports one or two resident text backends, aggregated cancellation/status and isolated model failures.
- Offline imports submit to both correction workers before waiting; live recording retains independent bounded asynchronous queues.
- Rejected/no-op requests also refresh panel status; model changes clear completed displays to avoid relabelling old results.

### Notes
- All 79 automatic tests passed. Two consecutive real dual imports reused both models, preserved source/evidence and exported matching results; 0.8B-only BF16 import also passed. Both layouts inspected at 900x700 and 720x600.
- Mixed-precision comparison peaked at 4241 MiB process CUDA allocation; requests overlapped about 5s. Concurrency was slower than standalone inference on this shared GPU.
- Same-prompt punctuation behavior: 0.8B BF16 5/12 and NF4 2/12, versus 2B application behavior 12/12 (one NF4 hard-cut rejection). Small-model quality is experimental; details in BENCHMARK_v1.3.0.md.
- v1.2.1 remains a rollback tag; dual-model work is isolated on codex/dual-qwen-comparison. No ASR model switch or new runtime environment was introduced.

## v1.4.0 - 2026-09-29

### Added
- Session-level writable window control (256–4096 Unicode characters, including 2048), persisted per SQLite store without unfreezing history.
- Read-only evidence replay CLI and regression coverage for offline ASR/LLM ordering.
- GPL-3.0-only source license and public README with installation, China model sources, dependency versions, hardware requirements and privacy guidance.

### Changed
- Long-window correction uses bounded 64-character targets, compact output and one conservative punctuation retry; generation budget is 192 tokens and batch timeout 90 seconds.
- Target lexical changes require alternate ASR support; copied context, omissions and unsupported rewrites are rejected. Valid targets can apply independently, with partial failures shown in the UI and audit.
- Launch and CUDA repair scripts prefer the project's existing .venv when present.

### Fixed
- Frequent long-input correction failures caused by truncated whole-window outputs. Failed retries still retain original evidence instead of implying complete correction.
- Legacy TXT filename collisions when sessions start within the same clock tick.

### Notes
- All 86 automated tests passed. Actual Qt single/dual layouts inspected at 900x700 and 720x600.
- Same 38-segment evidence replay increased punctuated segments from 7 to 28 and punctuation characters from 27 to 104; non-punctuation text remained unchanged. Eleven targets still failed content checks. Semantic punctuation accuracy remains unaccepted.
- A 60-second excerpt passed real Paraformer FP16 + Qwen2B BF16 offline import at window 2048: no dropped segments, matching UI/SQLite/TXT, unchanged input. No fresh dual-model/NF4 quality benchmark in this iteration.
- Public publication excludes recordings, transcripts, model weights, logs and local development history. See BENCHMARK_v1.4.0.md for measured limits.

## v1.4.1 - 2026-09-29

### Changed
- Update the local checkout from ycts-otree/LLM_ASR main at 8464a7f (v1.4.0), retaining the old local branch as a rollback point.
- Retain the existing G:\Python\Python_Environment\Python310\python.exe startup configuration.

### Notes
- All 86 automated tests passed; pip check reports no dependency conflicts. Real Qt sample import loaded ASR/Qwen, applied punctuation, dropped no segments and exported matching UI/TXT/SQLite content.
- The existing offscreen window-size assertion still fails; GUI sizing verification remains incomplete.
- User MP3, model weights, recordings and transcripts were preserved. No remote push was performed.
