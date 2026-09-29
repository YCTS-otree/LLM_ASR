"""Event-driven, bounded asynchronous patch pipeline with a deterministic mock."""
from dataclasses import dataclass
from typing import Protocol
import logging
import queue
import re
import threading
import time
from llm_protocol import BackendResult
from patches import RevisionEngine, PatchRejected
from evidence import ASREvidence
from transcript_store import TranscriptSnapshot

log = logging.getLogger('meeting_asr')
SYSTEM_PROMPT = ('Correct transcription only using evidence. Return replace/insert/delete JSON patches. '
                 'Preserve spoken content and style. Never modify the frozen prefix or invent speech.')


@dataclass(frozen=True)
class CorrectionContext:
    snapshot: TranscriptSnapshot
    evidence: ASREvidence
    system_prompt: str = SYSTEM_PROMPT
    readable_context: str = ''
    segments: tuple = ()
    readable_right: str = ''
    focused: bool = False
    punctuation_only: bool = False
    punctuation_mode: str = 'combined'
    glossary: tuple = ()


class ContextBuilder:
    def build(self, store, stable, punctuation_mode='combined', glossary=()):
        from pause_evidence import pause_hints
        snapshot = store.snapshot()
        boundaries = []
        offset = 0
        for segment in store.segments:
            end = offset + len(segment.current_text)
            if end > snapshot.window_start:
                evidence = store.evidence(segment.segment_id)
                boundaries.append(dict(id=segment.segment_id,
                    start=max(offset, snapshot.window_start)-snapshot.window_start,
                    end=end-snapshot.window_start, end_reason=evidence['end_reason'],
                    pause_after_ms=evidence.get('pause_after_ms'),
                    timing=pause_hints(evidence) if punctuation_mode!='baseline' else None,
                    best_text=evidence['best_text'],nbest=evidence.get('nbest',[])))
            offset = end
        return CorrectionContext(snapshot, stable.evidence,
                                 readable_context=snapshot.text[max(0, snapshot.window_start - 256):snapshot.window_start],
                                 segments=tuple(boundaries),punctuation_mode=punctuation_mode,glossary=tuple(glossary))


class LLMBackend(Protocol):
    def process(self, context: CorrectionContext) -> dict | BackendResult: ...


class MockLLMBackend:
    """Deterministic plumbing demo, NOT a real language/semantic correction model."""
    def process(self, context):
        snapshot = context.snapshot
        text, start = snapshot.writable_text, snapshot.window_start
        operations = []
        # Non-overlapping examples; no fake sentence boundary after max_duration.
        pattern = re.compile(r'(?<![A-Za-z0-9])(?:H170|PCIE)(?![A-Za-z0-9])|我们我们|，，|。。')
        for match in pattern.finditer(text):
            old = match.group()
            a, b = start + match.start(), start + match.end()
            if old == '我们我们':
                operations.append(dict(op='delete', start=a, end=a + 2, text=''))
            elif old in ('，，', '。。'):
                operations.append(dict(op='delete', start=a, end=a + 1, text=''))
            else:
                operations.append(dict(op='replace', start=a, end=b, text={'H170': 'HX170', 'PCIE': 'PCIe'}[old]))
            if len(operations) == 16:
                break
        return dict(base_revision=snapshot.revision, window_start=snapshot.window_start,
                    window_end=snapshot.window_end, operations=operations)


class CorrectionWorker(threading.Thread):
    """One running request + one latest pending request. Never blocks ASR for LLM."""
    def __init__(self, store, backend, changed, event=None, debounce=0):
        super().__init__(name='llm-patch-worker', daemon=True)
        self.store, self.backend, self.changed = store, backend, changed
        self.event = event or (lambda kind, data: None)
        self.requests = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.submit_lock = threading.Lock()
        self.revisions = RevisionEngine(store)
        self.debounce = debounce

    def submit(self, context):
        with self.submit_lock:
            if self.closed.is_set():
                raise RuntimeError('Correction worker is closed')
            try:
                self.requests.put_nowait(context)
            except queue.Full:
                try:
                    old = self.requests.get_nowait()
                except queue.Empty:
                    old = None  # Worker consumed it between Full and get_nowait.
                if old is not None:
                    self.requests.task_done()
                    self.event('LLM_REQUEST_COALESCED', {'base_revision': old.snapshot.revision})
                self.requests.put_nowait(context)

    def run(self):
        while not self.closed.is_set() or not self.requests.empty():
            try:
                context = self.requests.get(timeout=.1)
            except queue.Empty:
                continue
            # A small fixed debounce merges short adjacent segments without an
            # unbounded wait. Closing a session bypasses further debounce.
            deadline = time.monotonic() + self.debounce
            while not self.closed.is_set() and time.monotonic() < deadline:
                try:
                    latest = self.requests.get(timeout=max(.001, deadline - time.monotonic()))
                except queue.Empty:
                    break
                self.requests.task_done()
                self.event('LLM_REQUEST_COALESCED', {'base_revision': context.snapshot.revision})
                context = latest
            metadata = {}
            try:
                result = self.backend.process(context)
                if isinstance(result, BackendResult):
                    metadata = dict(result.metadata)
                    if result.error:
                        log.warning('LLM 校对失败 revision=%s result=%s detail=%s',
                                    context.snapshot.revision,result.error,metadata.get('detail',''))
                        self.event('LLM_REQUEST', dict(metadata, result=result.error))
                        continue
                    result = result.patch
                if not getattr(self.backend, 'enabled', True):
                    self.event('LLM_REQUEST', dict(metadata, result='RUNTIME_ERROR', detail='Disabled before application'))
                    continue
                previous = context.snapshot.revision
                revision = self.revisions.apply(result, context.snapshot,
                    source=getattr(self.backend, 'source', 'MOCK_LLM'), metadata=metadata or None)
                if metadata:
                    log.info('LLM 校对结束 revision=%s result=%s latency=%.2fs requests=%s',
                             previous,'APPLIED' if revision != previous else 'NO_CHANGE',
                             metadata.get('latency',0),metadata.get('request_count',1))
                    self.event('LLM_REQUEST', dict(metadata, result='APPLIED' if revision != previous else 'NO_CHANGE',
                                                   applied_revision=revision))
            except PatchRejected as exc:
                log.info('patch rejected revision=%s reason=%s', context.snapshot.revision, exc.code)
                if metadata:
                    self.event('LLM_REQUEST', dict(metadata, result='STALE' if exc.code == 'STALE_PATCH' else 'VALIDATION_REJECTED',
                                                   detail=exc.code))
            except Exception as exc:
                log.exception('correction worker failed')
                self.event('LLM_ERROR', {'base_revision': context.snapshot.revision, 'error': str(exc)})
            finally:
                try:
                    self.changed()  # Also refresh latency/status after rejected/no-op output.
                finally:
                    self.requests.task_done()

    def finish(self):
        with self.submit_lock:
            self.closed.set()
        self.join()


class MeetingPipeline:
    def __init__(self, store, changed=lambda: None, backend=None, punctuation_mode='combined', glossary=None):
        from pause_evidence import PUNCTUATION_MODES
        if punctuation_mode not in PUNCTUATION_MODES:raise ValueError('Invalid punctuation mode')
        self.punctuation_mode=punctuation_mode
        from glossary import load_terms
        try:self.glossary=tuple(load_terms() if glossary is None else glossary)
        except (OSError,ValueError):
            self.glossary=()
            store.record_event('GLOSSARY_UNAVAILABLE',{'detail':'术语库不可用，请检查JSON格式；本次未使用术语库'})
        from evidence import StabilityBuffer
        self.store, self.changed = store, changed
        self.stability = StabilityBuffer()
        self.context = ContextBuilder()
        selected = backend or MockLLMBackend()
        debounce = getattr(getattr(selected, 'config', None), 'debounce_seconds', 0)
        self.worker = CorrectionWorker(store, selected, changed, store.record_event, debounce)
        self.worker.start()

    def accept(self, evidence):
        stable = self.stability.push(evidence)
        with self.store.lock:
            self.store.append(stable,use_baseline=self.punctuation_mode!='pauses')
            context = self.context.build(self.store, stable,self.punctuation_mode,self.glossary)
        self.changed()
        if evidence.best_text and getattr(self.worker.backend, 'enabled', True) and getattr(self.worker.backend, 'ready', True):
            self.worker.submit(context)
        return stable

    def close(self):
        self.worker.finish()
