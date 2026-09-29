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

### Added
- Explicit Load model button: startup, model/precision edits and re-enabling correction no longer trigger automatic loading. Parameter edits remain pending until requested.
- Local ASS reference comparison tool with timestamp range selection and punctuation-insensitive character edit distance; reference text stays outside model inputs.

### Fixed
- Precision-dependent Paraformer FP16 CIF failure that omitted the first 15 seconds of a clean voiceover. AMP IndexError now retries the same audio once with FP32 and records actual precision/recovery reason.
- Offline verification now fails on ASR_FAILED events instead of treating zero queue drops and matching saved output as proof of completeness.
- Beginning a session with unloaded/changed model settings prompts for Load model without silently loading or capturing.

### Notes
- All 92 automated tests passed; actual single/dual Qt layouts inspected at 900x700 and 720x600.
- Same 60-second sample now covers all four 15-second segments with one audited FP32 recovery and zero ASR failures. Source unchanged; UI, SQLite and TXT match.
- Against separately supplied edited AI subtitles, character difference rate decreased from 37.73% to 9.76% by recovering missing speech. Current LLM did not further reduce lexical difference. Numeral/product-name formatting also contributes to this metric.
- Corrected v1.4.0's incomplete functional-test claim in its report. Meeting-quality acceptance is not met; bounded lexical-correction experiment showed no improvement and was not adopted. No model switch or reference-answer injection.

## v1.5.0 - 2026-09-29

### Added
- Optional DeepSeek HTTPS backend with an in-memory API key dialog, editable model ID, thinking toggle, redacted failures and no automatic network/auth retry.
- Independent CPU CT-Transformer baseline punctuation from ModelScope, available with LLM disabled or rejected; raw ASR remains immutable.
- Optional Qwen thinking mode and explicit distinction between lexical changes and punctuation/format-only patches.

### Changed
- New evidence revisits the complete editable window, including previous completed sentences; default UI window is2048 characters. Overlapping read context and disjoint128-character write targets prevent repeated concatenation.
- Contextual corrections, numeric/year/terminology normalization no longer require membership in ASR candidates. Full, target-associated candidates are supplied.
- Increased output/batch budgets for quality-oriented processing; retain revision/frozen/overlap checks and bounded edits. Non-duplicate pure lexical deletion and large rewrites are rejected.

### Fixed
- English initial capitalization by the punctuation model no longer causes loss of all punctuation in a segment; original lexical characters are preserved.
- API and thinking parameter changes remain staged until Load model; the small-window GUI was verified after adding controls.

### Notes
-102 automated tests passed; single/dual/API layouts and API settings were visually checked.
- Final60-second offline check: four complete segments, no ASR/punctuation failures or queue drops, matching UI/SQLite/TXT and unchanged source. Baseline punctuation23 marks, final25; punctuation count is not accuracy.
- Reference character difference remains9.76% before and after Qwen. Thinking probe exhausted4096 tokens without a final answer; thinking defaults off. Meeting-quality acceptance remains unmet.
- DeepSeek transport is mock-tested, not live-tested without a user API key. No fresh0.8B/dual quantization quality run. See BENCHMARK_v1.5.0.md.

## v1.5.1 - 2026-09-29

### Added
- Blank DeepSeek key input now loads DEEPSEEK.key from the current process working directory on explicit Load model. Accepts UTF-8 BOM and surrounding whitespace; manual input takes precedence.

### Fixed
- Clicking Load model rereads API credentials even when already configured; failed reload clears the previous credential and readiness.
- Missing, malformed or oversized key files produce content-free errors. The application never creates/rewrites the file or displays its contents; existing *.key Git exclusion remains in effect.

### Notes
-107 tests passed using temporary dummy credentials only. API settings GUI inspected. No real key was read and no live API request was made.

## v1.6.0 - 2026-09-29

### Added
- Configurable LLM target length, output-token budget and batch timeout; DeepSeek effort low/high/max, request timeout, non-thinking temperature and optional unchanged-answer punctuation retry. Parameters remain staged until Load model.
- Baseline-only, pause-only and combined punctuation strategies. Immutable low-energy/timestamp evidence carries explicitly approximate raw-word anchors or time-only hints.
- Editable runtime glossary.json with bounded contextual retrieval, bundled examples and no unconditional replacement. User glossary is excluded from Git.
- Narrow, audited written-number formatting for clear specifications and years, including2nm,8000mAh and锐龙9000系. Prompt includes unit casing and idiom-preservation rules.
- Request counts and token/normalization details in result-title tooltips and session metadata.

### Changed
- DeepSeek defaults to512-character targets instead of128 and skips optional no-change punctuation retries. Local targets remain128 by default. Both allow64–2048 characters with overlapping read context and disjoint write ranges.
- Focused answers produce minimal validated patches; large input targets no longer fail merely because the unchanged source text exceeds128 characters.

### Fixed
- Preserve baseline terminal punctuation at technical audio cuts; semantic boundaries are no longer rejected solely because the segment reached its duration limit.
- Repeated text does not shift an isolated punctuation edit into distant deletion/insertion operations.

### Notes
-119 tests passed; single/dual/API layouts and settings/glossary dialogs visually checked.
- Local six-case formatting probe passed6/6 with four rule-assisted cases, versus2/6 prompt-only. This is not a semantic-model accuracy gain.
-60-second real import: four segments, zero ASR/punctuation failures or drops, source unchanged and UI/SQLite/TXT equal. Sparse pause hints in this compressed sample; no live API latency measurement. See BENCHMARK_v1.6.0.md.

## v1.6.1 - 2026-09-29

### Fixed
- Operational logs now appear in both the console and rotating pipeline.log, without duplicate handlers. Debug transcript output remains file-only when explicitly enabled.
- DeepSeek requests log start, completion latency, token usage and sanitized failures; HTTP402 now explains insufficient API balance in the UI and logs. Other common HTTP failures have actionable explanations without response bodies or credentials.

### Notes
-121 automated tests passed, including console/file output and redacted402 handling. GUI layouts checked; no live API call or balance query was made.
