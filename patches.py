"""Strict patch protocol and validation. Offsets use Python Unicode indexing."""
from dataclasses import dataclass
import json


class PatchRejected(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Operation:
    op: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Patch:
    base_revision: int
    window_start: int
    window_end: int
    operations: tuple[Operation, ...]

    @classmethod
    def parse(cls, data):
        def strict_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise PatchRejected('MALFORMED_PATCH')
                value[key] = item
            return value
        try:
            if isinstance(data, str):
                if len(data) > 65536:
                    raise PatchRejected('SIZE_GUARD')
                data = json.loads(data, object_pairs_hook=strict_object)
            if type(data) is not dict or set(data) != {'base_revision', 'window_start', 'window_end', 'operations'}:
                raise PatchRejected('MALFORMED_PATCH')
            if any(type(data[k]) is not int or data[k] < 0 for k in ('base_revision', 'window_start', 'window_end')):
                raise PatchRejected('MALFORMED_PATCH')
            if type(data['operations']) is not list or len(data['operations']) > 256:
                raise PatchRejected('MALFORMED_PATCH')
            operations = []
            for op in data['operations']:
                if type(op) is not dict or set(op) != {'op', 'start', 'end', 'text'}:
                    raise PatchRejected('MALFORMED_PATCH')
                if (op['op'] not in ('replace', 'insert', 'delete') or type(op['text']) is not str or
                        type(op['start']) is not int or type(op['end']) is not int):
                    raise PatchRejected('MALFORMED_PATCH')
                # Reject unpaired surrogates: valid Python offsets, but invalid UTF-8 storage.
                op['text'].encode('utf-8')
                operations.append(Operation(**op))
            return cls(data['base_revision'], data['window_start'], data['window_end'], tuple(operations))
        except (TypeError, KeyError, ValueError, UnicodeError, RecursionError) as exc:
            if isinstance(exc, PatchRejected):
                raise
            raise PatchRejected('MALFORMED_PATCH') from exc


@dataclass(frozen=True)
class PatchLimits:
    max_operations: int = 256
    max_removed_chars: int = 1024
    max_inserted_chars: int = 1024
    max_operation_span: int = 128

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in vars(self).values()):
            raise ValueError('Patch limits must be non-negative integers')


class PatchValidator:
    def __init__(self, limits=None):
        self.limits = limits or PatchLimits()

    def validate(self, patch, snapshot, revision, text, frozen_boundary):
        if patch.base_revision != revision or snapshot.revision != revision:
            raise PatchRejected('STALE_PATCH')
        if (snapshot.text != text or snapshot.window_start != frozen_boundary or
                snapshot.window_end != len(text) or patch.window_start != snapshot.window_start or
                patch.window_end != snapshot.window_end):
            raise PatchRejected('WINDOW_MISMATCH')
        limits = self.limits
        if len(patch.operations) > limits.max_operations:
            raise PatchRejected('SIZE_GUARD')
        removed = inserted = 0
        ordered = sorted(patch.operations, key=lambda op: (op.start, op.end))
        previous = None
        for op in ordered:
            if not 0 <= op.start <= op.end <= len(text):
                raise PatchRejected('OFFSET_OUT_OF_RANGE')
            if op.start < frozen_boundary:
                raise PatchRejected('FROZEN_AREA')
            if op.start < snapshot.window_start or op.end > snapshot.window_end:
                raise PatchRejected('OUTSIDE_SNAPSHOT')
            if ((op.op == 'insert' and (op.start != op.end or not op.text)) or
                (op.op == 'delete' and (op.start == op.end or op.text)) or
                (op.op == 'replace' and (op.start == op.end or not op.text))):
                raise PatchRejected('INVALID_OPERATION')
            # Adjacent nonempty spans are defined. Insertions touching any other
            # operation endpoint are rejected to avoid ambiguous ordering.
            if previous and (op.start < previous.end or
                             (op.start == previous.end and
                              (op.start == op.end or previous.start == previous.end))):
                raise PatchRejected('OVERLAPPING_OPERATIONS')
            previous = op
            removed += op.end - op.start
            inserted += len(op.text)
            if op.end - op.start > limits.max_operation_span:
                raise PatchRejected('SIZE_GUARD')
        if removed > limits.max_removed_chars or inserted > limits.max_inserted_chars:
            raise PatchRejected('SIZE_GUARD')
        return tuple(ordered)


class RevisionEngine:
    """The only patch writer. Validation and commit share the store's lock."""
    def __init__(self, store, validator=None):
        self.store = store
        self.validator = validator or PatchValidator()

    def apply(self, data, snapshot, source='MOCK_LLM', metadata=None):
        if source not in {'MOCK_LLM', 'LOCAL_LLM', 'ONLINE_LLM', 'SYSTEM'}:
            raise PatchRejected('INVALID_SOURCE')
        with self.store.lock:
            try:
                # Parse even dataclasses, so manually constructed instances cannot
                # bypass schema checks.
                from dataclasses import asdict
                patch = Patch.parse(json.dumps(asdict(data)) if isinstance(data, Patch) else data)
                operations = self.validator.validate(patch, snapshot, self.store.revision,
                                                     self.store.canonical_text,
                                                     self.store.frozen_boundary)
                if not operations:
                    self.store.record_event('PATCH_NO_CHANGE', {'source': source, 'base_revision': patch.base_revision})
                    return self.store.revision
                return self.store._commit_patch(patch, operations, source, metadata)
            except PatchRejected as exc:
                self.store.record_event('PATCH_REJECTED', {'source': source, 'reason': exc.code,
                                                         'base_revision': snapshot.revision})
                raise
