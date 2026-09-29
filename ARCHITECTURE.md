# Contextual transcript infrastructure — v1.6.0

## Current behavior (supersedes historical version notes below)

The UI default writable window is 2048 characters. Every new segment revisits
ALL editable history, including finished sentences. Target limits are configurable
from64 to2048 characters (Qwen default128, DeepSeek512), preferring punctuation/Latin word boundaries. Read spans overlap
and include all other text in the window; write ownership does not overlap.
Frozen history remains immutable. Audio still uses maximum 15-second segments;
text context cannot recover audio missed by ASR.

Contextual lexical corrections and number/year/terminology normalization no
longer require exact N-best membership; full candidates are retained. Large
rewrites and non-duplicate pure deletions are rejected. Patch checks retain
revision/frozen/range/overlap checks, with 256 operations, 1024 inserted/removed
characters per batch, and 128 removed characters per operation. These bounds
are not proof that the model's proposed words are semantically correct.

An independent CPU CT-Transformer punctuation model seeds current_text in baseline
and combined modes; pauses-only mode seeds raw text. Both evidence forms stay
immutable. 20ms RMS measurements identify low-energy spans of at least120ms;
timestamp-aligned word hints are explicitly approximate and otherwise time-only.
Qwen optionally enables thinking. Output defaults to1024 tokens, configurable
to8192; batch timeout defaults600 seconds, configurable to1800.

DeepSeek uses the same target/patch flow through HTTPS chat completions. Its key
is used in memory only; when the UI key is blank, explicit Load model reads
DEEPSEEK.key relative to the process working directory (including reloads).
The application never creates/rewrites that file or displays its contents.
Redirects are blocked; errors expose codes only. No audio,
raw response or reasoning is logged/sent as additional data. Requests send
transcript/context/candidates, relevant terminology and selected pause hints.
Output defaults8192 tokens (256–32768 configurable), request timeout180 seconds
(5–600 configurable), bounded by the batch deadline. Thinking effort is
low/high/max; temperature is sent only without thinking. Unchanged-result
punctuation retries default off for API and on for local Qwen. Metadata tracks
request count, token usage, reasoning-token count and written-number normalization.
Network/auth failures are not automatically retried. Load model checks
configuration, not remote authentication. Earlier version notes below describe
historical candidate-only/short-tail policies, not current behavior.

Focused full-text answers pass lexical drift guards before deterministic, narrow
number/unit formatting. Minimal diffs are then validated with unchanged patch
limits; a large input target does not authorize a large replacement. Trimming
common edges prevents repeated text from shifting single punctuation edits.
An optional runtime glossary.json supplies at most8 matched reference entries;
it never substitutes text directly. The bundled example includes a user-supplied
phone name, so tests using it are terminology-assisted, not blind benchmarks.

## Scope and principles

1. ASR generates evidence; it does not define the final transcript.
2. LLM may read broader context, but may modify only the configured recent window (UI default 2048 Unicode characters; 256–4096 selectable).
3. Transcript outside the writable window is immutable.
4. Visual information is contextual evidence only and cannot directly modify transcript.
5. Every LLM modification must be represented as a validated and reversible patch.
6. Raw ASR results must always be preserved.
7. Local and online LLM backends must share the same logical interface.
8. Models may remain resident, but unnecessary inference must be avoided.

This version uses the existing offline Paraformer weights, RMS segmentation and
PyTorch runtime. Qwen3.5-2B and optional 0.8B provide local text correction. No streaming checkpoint,
neural VAD, second ASR, vision model, ONNX or NPU runtime is added. DeepSeek is optional.

## Data and concurrency

```text
Qt UI                         ModelLoader (startup, background)
  |                                |
  |                           Application ModelHost (resident Engine)
  |                                |
  +-- Session QThread: audio queue -> Engine -> ASREvidence -> StabilityBuffer
  |                                    |                         |
Capture thread                        N-best                 TranscriptStore
  |                                                             |
mic -> RF64 WAV -> segmenter -> bounded audio queue         ContextBuilder
  |                                                             |
overflow events                                       bounded latest request queue
                                                                |
                                                    CorrectionWorker / QwenLocalBackend
                                                                |
                                              Parser -> Validator -> RevisionEngine
                                                                |
                                                SQLite -> UI notification -> read latest
```

Each selected correction model has its own Store and correction worker within the session. The ASR model lives in
application-owned ModelHost across sessions. Loading never runs on the UI thread.
Changing device/precision explicitly replaces the resident model on the next
recording start. There is one ASR inference at a time; the independent Qwen worker
does not join or hold an ASR lock during generation. Both share GPU resources,
so contention can still increase inference latency. Store transactions use a shared RLock so
validation plus commit is atomic relative to ASR appends.

The LLM queue contains at most one pending context plus the running request.
Pending requests are coalesced to the latest snapshot and recorded in events.
Running stale responses are discarded, without automatic rebase/retry. Thus a
continuously changing transcript can temporarily prevent corrections; safety has
priority. Backend interface is `process(CorrectionContext) -> patch dict | BackendResult`.
Future network/model adapters must add bounded generation/timeout/cancellation;
the deterministic mock remains test-only, and session shutdown drains the workers.

## Audio time, finality and loss

AudioSegment has a unique ID, native captured sample start/end (end exclusive),
sample rate, computed seconds/duration, end reason and resampled 16 kHz audio.
The original sample coordinates are retained after resampling. They are offsets
into the captured WAV, not wall-clock or inference-completion timestamps.
Device overflow can lose an unknown number of samples before capture; the event
marks this uncertainty instead of fabricating a continuous physical timeline.

End reasons: `silence`, `max_duration`, `manual_stop`, `shutdown_flush`, `end_of_file`.
StabilityBuffer enforces chronological nonoverlap and preserves adjacent segment
relationships. Hard cuts never imply sentence-final punctuation. It emits one
stable observation per offline result, not partial streaming hypotheses.

Audio queue is bounded at 32 segments. WAV is written before queue insertion.
When full, drop the new *ASR queue item*, keep recording, report the skipped
segment's native sample range and WAV path in SQLite events and rotating logs.
No auto-recovery/backfill is implemented. `ASR_FAILED` likewise records source
audio ranges while later segments continue. PortAudio overflow is a distinct
`DEVICE_OVERFLOW` event; those missing samples cannot be recovered from WAV.
Disk/input failures still stop capture. RF64 avoids ordinary RIFF's4GB limit;
not every third-party player supports RF64. Audio is flushed regularly, but a
power failure can still lose buffered audio or damage an unfinished container.

## N-best and timestamps

`ParaformerAdapter` configures the already-installed `BeamSearchPara` and observes
its returned Hypothesis objects through an instance-local forward wrapper. It
does not edit site-packages. Supported installed version is pinned by requirements
to FunASR 1.2.7; an upstream upgrade requires re-verification of these internals.
Beam 1/3/5 can be configured independently from returned nbest, with nbest<=beam.
The adapter retains raw decoder scores, never interprets them as accuracy or
calibrated confidence (`confidence=None`).

Each hypothesis is detokenized through the baseline non-timestamp postprocessor.
Only top 1 becomes best_text. The upstream raw result list is preserved separately,
including timestamp-mode formatting. Timestamps are requested with
`pred_timestamp=True`, validated as local intervals, converted from milliseconds
to seconds, and exposed globally by adding segment.start_time. Alignment endpoint
validation allows 150ms tolerance beyond the audio duration, because these are
model estimates. Do not assume one
timestamp per Python character: tokenization/postprocessing can differ. Missing or
invalid timestamps remain empty with a warning; timestamp-enabled failures retry
text-only. Both attempts count toward recorded inference latency. Nonfinite raw
JSON numbers are preserved as tagged strings rather than breaking persistence.

## Canonical text and immutable window

Canonical text is the exact concatenation of segment.current_text in capture
order, with **no implicit separator**. ASR does not insert guessed punctuation at
hard cuts. Offsets use Python `str` indexing (Unicode code points), never bytes,
UTF-16 units, grapheme clusters or tokenizer tokens.

Store maintains:

```python
new_boundary = max(old_boundary, len(canonical_text) - 1024, 0)
```

Because all accepted edits are at or after the old boundary, the frozen prefix
and its character positions never change. Deletions may leave **fewer** than 1024
writable characters. This is intentional: frozen text must never re-enter the
window. Insertions at the boundary or at EOF are permitted.

Offset mapping assigns a segment boundary to the following nonempty segment,
and EOF to the last segment. A cross-segment replacement puts new text into the
first affected segment and removes covered text from following segments. Empty
segments remain in the store to retain their raw evidence and timing. An insertion
at a boundary follows the same mapping. Multiple operations are applied in
descending offset order using one original snapshot.

## Patch contract

```json
{
  "base_revision": 3,
  "window_start": 0,
  "window_end": 4,
  "operations": [
    {"op": "replace", "start": 0, "end": 4, "text": "HX170"}
  ]
}
```

All four fields on each operation are required. For insert, start==end and text
is nonempty; for delete, start<end and text is empty; replace requires a nonempty
span and replacement. Unknown fields, duplicate JSON keys, boolean offsets,
unpaired Unicode surrogates and malformed schemas are rejected.

Patch base_revision must equal current Store revision and issued snapshot
revision. The snapshot text and exact window bounds must still match Store;
operations must be inside that snapshot and at/after frozen_boundary. Nonempty
spans may be adjacent but not overlap. Insertions at another operation's endpoint
are rejected as ambiguous. Limits are configurable via PatchLimits; defaults are
32 operations, 256 total removed characters, 256 inserted characters and 128 removed
characters per operation. Limits are technical guards, not semantic correctness.

An empty operations list records no-change without increasing revision. Every
ASR append and accepted nonempty patch increments revision, even if a patch's
replacement happens to equal existing text. Rejected patches leave text/revision
unchanged and record the rejection code in events.

## Persistence and audit

SQLite is the source of truth, with WAL and synchronous=FULL. Tables:

- evidence: immutable serialized ASREvidence, indexed by segment ID.
- segments: immutable raw_asr_text plus current_text, time and creation/modification revisions.
- metadata: current revision and monotonic frozen boundary.
- revisions: timestamp, source, type, previous revision and append/patch information.
- events: overflow, failed ASR, coalesced requests, no-change and rejected patches.

Patch history retains original operations, removed/replacement text and before/
after text for each affected segment. These support future inverse operations and
audit without saving a full meeting snapshot on every revision. Undo UI/API is
not implemented; any future undo must respect the then-current frozen boundary.
Evidence/raw text/history have database triggers rejecting destructive updates.
TXT is atomically exported on normal/error session cleanup and can be regenerated
with `TranscriptStore(path).export_txt(output_path)`. After a crash, reopen SQLite;
do not delete its accompanying WAL/SHM files while a session is active.

Network inference occurs only when DeepSeek is selected. Models/logs/audio/databases are
excluded from Git. Transcript/audit history grows on disk intentionally; rotating
operational logs are bounded. UI currently rebuilds canonical text on updates;
very long meetings may eventually need incremental rendering and disk retention
controls. Multi-hour power, heat and fault recovery testing remains future work.

## Local Qwen runtime and scheduling

Window owns a CorrectionModels group across sessions. LocalModelLoader loads its selected backends after
ASR startup on a QThread; recording is available while the local LLM is loading.
Missing dependencies/weights or an LLM load error do not disable ASR. Disabling
correction cancels a running request at the next generation step and suppresses
new requests, retaining weights until application exit. Segments arriving while
disabled/loading are stored normally; no automatic backfill is scheduled.

Transformers' official Qwen3_5ForCausalLM loads the text submodel from official
multimodal safetensors without a vision tower. All runtime loads are local-only;
download_qwen.py is a separate explicit opt-in tool. BF16 is default; optional
bitsandbytes NF4 uses BF16 compute. No global FunASR package edits are needed.

The correction worker has a fixed 0.35s debounce and one pending latest snapshot.
Coalescing preserves all ASR segments in Store while only retaining the newest
request. A later ASR append invalidates the running request's snapshot; no silent
rebasing occurs. LLM exceptions/timeout/OOM/invalid output become audited no-op
outcomes. Timeout is cooperative between generation steps, not a hard CUDA-kernel
deadline. A driver hang cannot be forcefully interrupted within this process.

Context contains up to1024 writable Python characters, up to256 frozen read-only
characters, and latest raw ASR Top1/N-best/decoder scores/end_reason. It is not a
token streaming pipeline. max_duration explicitly means the segment may continue.
Greedy generation disables thinking, caps output at128 tokens and input at4096.
Prompt version is qwen_asr_correction_v6 in llm_protocol.py; rules live in prompts/.

## Model-facing patch protocols

Relative operations include op/start/end/text/old_text. start/end count Python
characters from writable_window[0], and exact old_text matching is required.
The translator adds window_start before invoking the existing Patch parser.

Anchored replace/delete use op/old_text/new_text; insert uses op/anchor/position/text.
The anchor must occur exactly once inside the writable window, including overlapping
matches. No fuzzy matching, inferred occurrence number or offset repair is allowed.
The default is anchored after real-model relative-offset failures. Canonical Patch
validation and the atomic revision/snapshot/frozen checks above remain unchanged.

Output must be a single complete JSON object with base_revision and operations.
Only surrounding whitespace is stripped. Fences, prose, duplicate keys, nonfinite
numbers, unknown fields and malformed/truncated JSON are rejected. BackendResult
carries either a canonical patch or failure plus metadata. Each LLM_REQUEST event
records the model, dtype, protocol, prompt version, snapshot, tokens, time and result;
accepted patch history additionally records the same request metadata.

The model receives explicit data/instruction separation, but this alone is not a
security guarantee. The validator limits where/how much it can edit, not semantic
truth. Current development cases demonstrate missed corrections; earlier prompt
variants also produced incorrect valid patches. Do not infer production accuracy
from successful JSON or a small test suite. MockLLMBackend remains test-only.

## v1.2.1 punctuation and offline input

The same correction request restores punctuation without acoustic N-best support
while lexical changes still require evidence. Fillers, wording and existing spaces
are preserved. Two generic anchored examples clarify output shape and unfinished
technical splits. ContextBuilder supplies the last 16 overlapping segment boundaries
using window-relative offsets, end_reason and pause_after_ms. The latter estimates
trailing below-threshold audio in existing 100 ms RMS blocks, not phonetic silence.
pause_before_ms is not inferred. Older evidence without a pause remains readable.

The translator validates the original operations before deterministic SequenceMatcher
diff (autojunk disabled), removes equal regions, and validates the resulting operations
again. The original 128-character span / 256-character aggregate / 32-operation limits
remain unchanged. Large window replacements cannot bypass limits through minimization.
Pure punctuation becomes insertions, preserving each segment's words and evidence.
Basic punctuation inserted exactly at a boundary belongs to the preceding nonempty
segment; other boundary insertions retain the existing next-segment behavior.

At max_duration, newly added terminal .?! or Chinese equivalents cause the entire
request to be rejected with HARD_CUT_TERMINATOR, audited as VALIDATION_REJECTED.
This conservative policy defers even a potentially complete sentence until further
context; it does not strip punctuation silently. Interior punctuation remains allowed.
Snapshot, revision and frozen-boundary validation still apply at commit time.

A decrease in counts of 嗯/啊/呃 rejects the entire request as ORAL_FILLER_REMOVED.
This narrow conservative protection follows an observed live failure; it can also
reject genuine duplicate artifacts and does not prove general semantic fidelity.

FileCapture reads a local file through existing SoundFile in 100 ms blocks, mixes
channels to mono, preserves the source, and writes a separate PCM16 RF64 archive.
It reuses Segmenter and resampling. Unlike microphone capture, it waits on a full
bounded queue and never invokes the drop policy. An abort event frees a blocked
producer if the consumer fails. Normal stop drains already queued work. EOF is
recorded as end_of_file; user stop and shutdown retain their own reasons.

File sessions wait for local model initialization (falling back to ASR on failure)
and wait for each correction request before appending the next ASR segment. This
avoids fast file input freezing text before correction. Microphone scheduling remains
asynchronous. UI import does not require a microphone, is mutually exclusive with
capture, and exports the same SQLite/TXT artifacts. Progress measures audio frames
read, not completed ASR/LLM work. No codec, model or runtime was installed.

## v1.3.0 model choice and comparison

qwen_spec.py pins official identities, revisions, local directories and weight
SHA256 for 2B and 0.8B. LocalLLMConfig carries model_size and QwenLocalBackend
records the actual selected identity. download_qwen.py --model explicitly selects
an official download; application loading stays local-only and text-only.

CorrectionModels owns one or two independent QwenLocalBackend instances and
aggregates load/status/enable/cancel/close. Models load sequentially to limit peak
initialization pressure, then remain resident. One failed load does not prevent
attempting the other. Changing the selection or precision between sessions closes
the previous group; repeated sessions reuse it. No shared model weights or locks
serialize the two backend workers. Both schedule work on the same CUDA device;
request concurrency does not guarantee simultaneous GPU kernels or speedup.

Session captures/reads audio and runs Paraformer once per segment. The resulting
immutable ASREvidence (same ID, audio span and N-best) fans out to independent
Store/MeetingPipeline pairs. Each has its own revision, frozen boundary, context,
correction queue and worker. A patch or error on one side cannot edit the other
store. Both databases reference one recording and share comparison_id for audit.
Exports use _2b and _0_8b suffixes; original evidence is duplicated intentionally
so either database remains independently inspectable.

Offline sessions submit the segment to both workers before joining either queue;
this permits concurrent inference while preserving every segment's correction
opportunity. Live capture never joins the workers per segment, retaining each
worker's bounded latest-request policy. Thus live scheduling can differ by model.
The two contexts also evolve independently after their first patch: this compares
two transcription pipelines, not identical subsequent model inputs. Standalone
punctuation cases provide identical-input model quality comparisons separately.

The UI uses layout-managed equal-width text panes, model selectors and separate
precision controls. Default comparison is 2B NF4 plus 0.8B BF16, based on measured
0.8B NF4 degradation; single-model default remains 2B BF16. Panel latency/result
comes from the latest request in that panel's session database, including rejected
or no-op requests. Selectors are disabled during loading or capture/import.

## v1.4.0 bounded correction and configurable window

A session fixes its writable-window size before input starts. Both comparison
stores receive the same size, recorded in SESSION_START and a durable SQLite
configuration table. Reopening a store preserves its configured size. Increasing
it never moves an existing frozen boundary backwards. Patch size limits remain
32 operations, 256 removed/inserted characters and 128 characters per operation.

For windows longer than 64 characters, focused_correction selects the latest
segment and up to 64 characters of the previous unfinished tail, partitions this
range into targets of at most 64 characters, and supplies 256 characters of
read-only left context and up to 96 characters of right context. A target is an
immutable snapshot prefix with original absolute offsets, never a live store.
This is a correction-permission window, not an exhaustive sweep of all older
text. Short windows retain the existing anchored/relative protocol.

Focused outputs contain only base_revision and target text. They are converted
to exact, minimal patches and validated before merging. Non-punctuation changes
require an exact alternate N-best span and at most four changed lexical
characters; omission/extension of the target is rejected. This deliberately
trades some contextual lexical correction recall for fidelity. The small-window
protocol retains its previous lexical policy. Oral fillers and hard-cut ending
guards remain enforced. Identical boundary insertions are deduplicated; genuine
conflicts are rejected, not guessed. All final operations are revalidated against
the original full snapshot by RevisionEngine.

A failed focused target, or a long no-op target, gets at most one conservative
punctuation-only retry. Both attempts share a 90-second batch deadline and
cancellation. The 192-token generation cap is per attempt. Successful targets
can be applied while failed targets retain original text. Per-target attempts,
errors, aggregate tokens, latency and partial_failures are audited; the UI
explicitly indicates incomplete correction. A valid patch is not a semantic
quality guarantee.

Offline Session already joins the correction workers after each ASR segment;
the next segment cannot overtake pending correction. A regression test asserts
ASR 0 -> LLM 0 -> ASR 1 -> LLM 1 with a deliberately slower backend. Microphone
capture remains asynchronous with bounded queues. replay_evidence reads old
ASR evidence through a read-only connection and writes a separate result store;
it never changes the source session or implies a new ASR measurement.

## v1.4.1 explicit load and completeness

Window stages model/precision selections without touching resident weights.
Load model applies the pending correction group and asynchronously loads/reuses
ASR then enabled local models. Beginning capture/import requires matching,
ready models; changing only output directory, threshold or writable window does
not require weight reload. Re-enabling an unloaded model does not start loading.

An AMP IndexError during ASR retries the same samples and resident model once
with CUDA autocast disabled. Evidence records inference_precision and
fallback_reason, and Session writes ASR_PRECISION_FALLBACK. Persistent errors
remain ASR_FAILED. Offline verification now rejects any ASR_FAILED event.
Reference subtitle comparison is evaluation-only and never changes prompts or
transcripts. A successful application test is not speech-quality acceptance.
