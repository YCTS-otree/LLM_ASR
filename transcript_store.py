"""Durable segment transcript and append-only audit history, backed by SQLite."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import threading
import math

WRITABLE_WINDOW_CHARS = 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def json_text(value):
    def safe(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {'nonfinite_raw_value': repr(item)}
        if isinstance(item, dict):
            return {key: safe(v) for key, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(v) for v in item]
        return item
    return json.dumps(safe(value), ensure_ascii=False, allow_nan=False, separators=(',', ':'))


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    start_time: float
    end_time: float
    raw_asr_text: str
    current_text: str
    asr_evidence_ref: str
    revision_created: int
    revision_modified: int


@dataclass(frozen=True)
class TranscriptSnapshot:
    revision: int
    text: str
    window_start: int
    window_end: int

    @property
    def writable_text(self):
        return self.text[self.window_start:self.window_end]


class TranscriptStore:
    def __init__(self, path=':memory:', window_chars=None):
        if window_chars is not None and (type(window_chars) is not int or not 256 <= window_chars <= 4096):
            raise ValueError('Writable window must be 256..4096 Unicode characters')
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1),
                revision INTEGER NOT NULL, frozen_boundary INTEGER NOT NULL);
            INSERT OR IGNORE INTO metadata VALUES(1,0,0);
            CREATE TABLE IF NOT EXISTS configuration (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (segment_id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS segments (ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT UNIQUE NOT NULL REFERENCES evidence(segment_id),
                start_time REAL NOT NULL, end_time REAL NOT NULL, raw_asr_text TEXT NOT NULL,
                current_text TEXT NOT NULL, asr_evidence_ref TEXT NOT NULL,
                revision_created INTEGER NOT NULL, revision_modified INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions (revision_id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL, type TEXT NOT NULL, source TEXT NOT NULL,
                previous_revision INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (event_id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS protect_raw_asr BEFORE UPDATE OF raw_asr_text ON segments
                BEGIN SELECT RAISE(ABORT, 'raw_asr_text is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS protect_segment_delete BEFORE DELETE ON segments
                BEGIN SELECT RAISE(ABORT, 'raw segment is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS protect_evidence_update BEFORE UPDATE ON evidence
                BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS protect_evidence_delete BEFORE DELETE ON evidence
                BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS protect_history_update BEFORE UPDATE ON revisions
                BEGIN SELECT RAISE(ABORT, 'history is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS protect_history_delete BEFORE DELETE ON revisions
                BEGIN SELECT RAISE(ABORT, 'history is immutable'); END;
        ''')
        saved = self.db.execute("SELECT value FROM configuration WHERE key='window_chars'").fetchone()
        self._window_chars = window_chars if window_chars is not None else (saved[0] if saved else WRITABLE_WINDOW_CHARS)
        self.db.execute("INSERT OR REPLACE INTO configuration VALUES('window_chars',?)", (self._window_chars,))
        self.db.commit()

    @property
    def window_chars(self):
        return self._window_chars

    @property
    def revision(self):
        with self.lock:
            return self.db.execute('SELECT revision FROM metadata WHERE id=1').fetchone()[0]

    @property
    def frozen_boundary(self):
        with self.lock:
            return self.db.execute('SELECT frozen_boundary FROM metadata WHERE id=1').fetchone()[0]

    @property
    def segments(self):
        with self.lock:
            rows = self.db.execute('SELECT segment_id,start_time,end_time,raw_asr_text,current_text,'
                                   'asr_evidence_ref,revision_created,revision_modified FROM segments ORDER BY ordinal')
            return tuple(TranscriptSegment(**dict(row)) for row in rows)

    @property
    def canonical_text(self):
        with self.lock:
            return ''.join(row[0] for row in self.db.execute('SELECT current_text FROM segments ORDER BY ordinal'))

    def snapshot(self):
        with self.lock:
            text = self.canonical_text
            return TranscriptSnapshot(self.revision, text, self.frozen_boundary, len(text))

    def offset_to_segment(self, offset):
        """Boundary belongs to the next nonempty segment; EOF to the last segment."""
        with self.lock:
            segments = self.segments
            if type(offset) is not int or not 0 <= offset <= len(self.canonical_text) or not segments:
                raise ValueError('Offset out of range')
            start = 0
            for segment in segments:
                end = start + len(segment.current_text)
                if offset < end:
                    return segment.segment_id, offset - start
                start = end
            return segments[-1].segment_id, len(segments[-1].current_text)

    def segment_to_offset(self, segment_id, local_offset):
        with self.lock:
            start = 0
            for segment in self.segments:
                if segment.segment_id == segment_id:
                    if type(local_offset) is not int or not 0 <= local_offset <= len(segment.current_text):
                        raise ValueError('Local offset out of range')
                    return start + local_offset
                start += len(segment.current_text)
            raise KeyError(segment_id)

    def _advance(self, revision, new_length):
        boundary = max(self.frozen_boundary, new_length - self.window_chars, 0)
        self.db.execute('UPDATE metadata SET revision=?, frozen_boundary=? WHERE id=1', (revision, boundary))
        return boundary

    def _history(self, revision, kind, source, data):
        self.db.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?)',
                        (revision, now(), kind, source, revision - 1, json_text(data)))

    def append(self, stable):
        evidence = stable.evidence
        initial = evidence.punctuated_text if evidence.punctuated_text is not None else evidence.best_text
        with self.lock, self.db:
            old_revision = self.revision
            old_boundary = self.frozen_boundary
            old_length = len(self.canonical_text)
            revision = old_revision + 1
            self.db.execute('INSERT INTO evidence VALUES(?,?)', (evidence.segment_id, json_text(evidence.to_dict())))
            self.db.execute('INSERT INTO segments(segment_id,start_time,end_time,raw_asr_text,current_text,'
                            'asr_evidence_ref,revision_created,revision_modified) VALUES(?,?,?,?,?,?,?,?)',
                            (evidence.segment_id, evidence.start_time, evidence.end_time, evidence.best_text,
                             initial, evidence.segment_id, revision, revision))
            boundary = self._advance(revision, old_length + len(initial))
            self._history(revision, 'ASR_APPEND', 'ASR', {
                'segment_id': evidence.segment_id, 'offset': old_length, 'before': '', 'after': initial,
                'baseline_punctuation': evidence.punctuated_text is not None,
                'end_reason': evidence.end_reason, 'hard_cut': stable.hard_cut,
                'follows_hard_cut': stable.follows_hard_cut, 'previous_segment_id': stable.previous_segment_id,
                'frozen_before': old_boundary, 'frozen_after': boundary})
            return revision

    def _commit_patch(self, patch, operations, source, metadata=None):
        # Called only by RevisionEngine with its validation/commit lock held.
        with self.lock, self.db:
            segments = self.segments
            texts = [s.current_text for s in segments]
            original = list(texts)
            before_text = ''.join(texts)
            changes = []
            for op in reversed(operations):
                if not segments:
                    from patches import PatchRejected
                    raise PatchRejected('NO_SEGMENT')
                # Ranges are measured in the request's original canonical text.
                # Descending application leaves all earlier offsets unchanged.
                position = 0
                owner = None
                for i, text in enumerate(texts):
                    end = position + len(text)
                    # Punctuation inserted exactly at a boundary belongs to the
                    # preceding nonempty segment; other insertions keep old rules.
                    if (owner is None and text and op.start == op.end == end and op.text
                            and all(c in '，。？！：；,.?!:;' for c in op.text)):
                        owner = i
                    if owner is None and (op.start < end or i == len(texts) - 1):
                        owner = i
                    if op.start < end and op.end > position:
                        a, b = max(0, op.start - position), min(len(text), op.end - position)
                        texts[i] = text[:a] + (op.text if i == owner else '') + text[b:]
                    elif op.start == op.end and i == owner:
                        local = op.start - position
                        texts[i] = text[:local] + op.text + text[local:]
                    position = end
                changes.append({'op': op.op, 'start': op.start, 'end': op.end,
                                'before': before_text[op.start:op.end], 'after': op.text})
            new_text = ''.join(texts)
            boundary_before = self.frozen_boundary
            if new_text[:boundary_before] != before_text[:boundary_before]:
                raise AssertionError('Frozen prefix changed')
            revision = self.revision + 1
            affected = []
            for segment, old, new in zip(segments, original, texts):
                if old != new:
                    self.db.execute('UPDATE segments SET current_text=?,revision_modified=? WHERE segment_id=?',
                                    (new, revision, segment.segment_id))
                    affected.append({'segment_id': segment.segment_id, 'before': old, 'after': new,
                                     'previous_revision_modified': segment.revision_modified})
            boundary = self._advance(revision, len(new_text))
            self._history(revision, 'LLM_PATCH', source, {
                'base_revision': patch.base_revision, 'window_start': patch.window_start,
                'window_end': patch.window_end, 'operations': [asdict(op) for op in operations],
                'changes': list(reversed(changes)), 'segments': affected,
                'frozen_before': boundary_before, 'frozen_after': boundary,
                'request_metadata': metadata})
            return revision

    def record_event(self, kind, data):
        with self.lock, self.db:
            self.db.execute('INSERT INTO events(timestamp,type,data) VALUES(?,?,?)', (now(), kind, json_text(data)))

    def history(self):
        with self.lock:
            return [dict(row, data=json.loads(row['data'])) for row in
                    self.db.execute('SELECT * FROM revisions ORDER BY revision_id')]

    def events(self):
        with self.lock:
            return [dict(row, data=json.loads(row['data'])) for row in
                    self.db.execute('SELECT * FROM events ORDER BY event_id')]

    def last_event(self, kind):
        with self.lock:
            row = self.db.execute('SELECT data FROM events WHERE type=? ORDER BY event_id DESC LIMIT 1', (kind,)).fetchone()
            return json.loads(row[0]) if row else None

    def evidence(self, segment_id):
        with self.lock:
            row = self.db.execute('SELECT data FROM evidence WHERE segment_id=?', (segment_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def export_txt(self, path):
        with self.lock:
            path = Path(path)
            temporary = path.with_suffix(path.suffix + '.tmp')
            with temporary.open('w', encoding='utf-8') as output:
                output.write(self.canonical_text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)

    def close(self):
        with self.lock:
            self.db.close()
