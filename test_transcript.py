import json
import sqlite3
import tempfile
import unittest
import random
from unittest.mock import patch
from pathlib import Path
from evidence import ASREvidence, StabilityBuffer
from transcript_store import TranscriptStore
from patches import Patch, PatchLimits, PatchValidator, PatchRejected, RevisionEngine


def add(store, text, identifier=None):
    identifier = identifier or str(store.revision + 1)
    ev = ASREvidence(identifier, float(store.revision), float(store.revision + 1), text)
    return store.append(StabilityBuffer().push(ev))


def patch_for(snapshot, *operations):
    return dict(base_revision=snapshot.revision, window_start=snapshot.window_start,
                window_end=snapshot.window_end, operations=list(operations))


def op(kind, start, end, text=''):
    return dict(op=kind, start=start, end=end, text=text)


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.store = TranscriptStore()
        self.engine = RevisionEngine(self.store)

    def tearDown(self):
        self.store.close()

    def apply(self, *operations):
        snapshot = self.store.snapshot()
        return self.engine.apply(patch_for(snapshot, *operations), snapshot)

    def rejected(self, code, *operations):
        snapshot = self.store.snapshot()
        original = self.store.canonical_text
        revision = self.store.revision
        with self.assertRaisesRegex(PatchRejected, code):
            self.engine.apply(patch_for(snapshot, *operations), snapshot)
        self.assertEqual(self.store.canonical_text, original)
        self.assertEqual(self.store.revision, revision)

    def test_window_small_exact_and_large(self):
        add(self.store, '中' * 100)
        self.assertEqual(self.store.snapshot().window_start, 0)
        add(self.store, 'a' * 924)
        self.assertEqual(self.store.snapshot().window_start, 0)
        add(self.store, '😀e\u0301')
        self.assertEqual(self.store.snapshot().window_start, 3)
        self.assertEqual(len(self.store.snapshot().writable_text), 1024)

    def test_segment_mapping_and_cross_segment_replacement(self):
        add(self.store, '中文H', 'a')
        add(self.store, '170接口😀', 'b')
        self.assertEqual(self.store.offset_to_segment(3), ('b', 0))
        self.assertEqual(self.store.segment_to_offset('b', 3), 6)
        self.apply(op('replace', 2, 6, 'HX170'))
        self.assertEqual(self.store.canonical_text, '中文HX170接口😀')
        self.assertEqual(self.store.segments[0].raw_asr_text, '中文H')
        self.assertEqual(self.store.segments[1].raw_asr_text, '170接口😀')
        self.assertEqual(self.store.segments[0].current_text, '中文HX170')
        self.assertEqual(self.store.segments[1].current_text, '接口😀')

    def test_frozen_boundary_midsegment_and_monotonic_after_deletion(self):
        add(self.store, '中' * 1100)
        self.assertEqual(self.store.offset_to_segment(76), ('1', 76))
        self.apply(op('delete', 1000, 1100))
        self.assertEqual(self.store.frozen_boundary, 76)
        self.assertEqual(len(self.store.snapshot().writable_text), 924)
        self.rejected('FROZEN_AREA', op('replace', 75, 76, '英'))
        self.rejected('FROZEN_AREA', op('insert', 75, 75, '英'))
        add(self.store, 'x' * 200)
        self.assertEqual(self.store.frozen_boundary, 176)

    def test_delete_entire_window_does_not_unfreeze(self):
        add(self.store, 'x' * 1100)
        snapshot = self.store.snapshot()
        engine = RevisionEngine(self.store, PatchValidator(PatchLimits(32, 1024, 256, 1024)))
        engine.apply(patch_for(snapshot, op('delete', 76, 1100)), snapshot)
        self.assertEqual(self.store.frozen_boundary, 76)
        self.assertEqual(self.store.snapshot().writable_text, '')
        self.rejected('FROZEN_AREA', op('replace', 0, 1, 'a'))
        self.apply(op('insert', 76, 76, '新'))
        self.assertEqual(self.store.canonical_text, 'x' * 76 + '新')

    def test_replace_insert_delete_and_unicode(self):
        add(self.store, '讨论H170😀PCIE')
        self.apply(op('replace', 2, 6, 'HX170'), op('replace', 7, 11, 'PCIe'))
        self.assertEqual(self.store.canonical_text, '讨论HX170😀PCIe')
        self.apply(op('insert', 12, 12, '。'))
        self.apply(op('delete', 7, 8))
        self.assertEqual(self.store.canonical_text, '讨论HX170PCIe。')

    def test_adjacent_replacements_use_snapshot_offsets(self):
        add(self.store, 'abcd')
        self.apply(op('replace', 0, 2, 'LONG'), op('replace', 2, 4, 'Z'))
        self.assertEqual(self.store.canonical_text, 'LONGZ')

    def test_out_of_range(self):
        add(self.store, 'abcd')
        for operation in [op('delete', -1, 1), op('delete', 2, 5), op('replace', 3, 2, 'a')]:
            self.rejected('OFFSET_OUT_OF_RANGE', operation)

    def test_stale_response_never_overwrites_new_asr(self):
        add(self.store, 'H170')
        snapshot = self.store.snapshot()
        add(self.store, '新内容')
        with self.assertRaisesRegex(PatchRejected, 'STALE_PATCH'):
            self.engine.apply(patch_for(snapshot, op('replace', 0, 4, 'HX170')), snapshot)
        self.assertEqual(self.store.canonical_text, 'H170新内容')
        self.assertEqual(self.store.revision, 2)

    def test_overlap_and_ambiguous_insertions(self):
        add(self.store, 'abcde')
        for operations in [
            [op('replace', 0, 3, 'x'), op('delete', 2, 4)],
            [op('insert', 1, 1, 'x'), op('insert', 1, 1, 'y')],
            [op('insert', 1, 1, 'x'), op('replace', 1, 2, 'y')],
            [op('delete', 0, 2), op('insert', 2, 2, 'y')]]:
            self.rejected('OVERLAPPING_OPERATIONS', *operations)

    def test_forged_window_rejected(self):
        add(self.store, 'abcd')
        snapshot = self.store.snapshot()
        data = patch_for(snapshot, op('replace', 0, 1, 'x'))
        data['window_end'] += 1
        with self.assertRaisesRegex(PatchRejected, 'WINDOW_MISMATCH'):
            self.engine.apply(data, snapshot)

    def test_size_guard(self):
        add(self.store, 'a' * 500)
        self.rejected('SIZE_GUARD', op('delete', 0, 129))
        self.rejected('SIZE_GUARD', op('insert', 0, 0, 'b' * 1025))

    def test_invalid_operation_shapes(self):
        add(self.store, 'abcd')
        for operation in [op('insert', 0, 1, 'x'), op('delete', 0, 1, 'x'),
                          op('replace', 1, 1, 'x'), op('delete', 1, 1), op('replace', 0, 1)]:
            self.rejected('INVALID_OPERATION', operation)

    def test_malformed_patch_and_duplicate_keys(self):
        for data in ['not json', '{}', '[]', '{"base_revision":1,"base_revision":2}',
                     dict(base_revision=True, window_start=0, window_end=1, operations=[]),
                     dict(base_revision=0, window_start=0, window_end=1,
                          operations=[op('replace', 0.0, 1, 'x')])]:
            with self.assertRaises(PatchRejected):
                Patch.parse(data)

    def test_history_has_before_after_source_and_raw_is_immutable(self):
        add(self.store, 'H170', 'segment')
        self.apply(op('replace', 0, 4, 'HX170'))
        self.assertEqual(self.store.revision, 2)
        history = self.store.history()
        self.assertEqual(history[0]['type'], 'ASR_APPEND')
        self.assertEqual(history[1]['source'], 'MOCK_LLM')
        self.assertEqual(history[1]['previous_revision'], 1)
        self.assertEqual(history[1]['data']['changes'][0]['before'], 'H170')
        self.assertEqual(history[1]['data']['changes'][0]['after'], 'HX170')
        self.assertEqual(history[1]['data']['segments'][0]['segment_id'], 'segment')
        self.assertEqual(self.store.evidence('segment')['best_text'], 'H170')
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("UPDATE segments SET raw_asr_text='changed'")
        self.store.db.rollback()

    def test_empty_patch_no_revision(self):
        add(self.store, '原文')
        self.apply()
        self.assertEqual(self.store.revision, 1)

    def test_history_write_failure_rolls_back_entire_patch(self):
        add(self.store, 'H170')
        with patch.object(self.store, '_history', side_effect=sqlite3.OperationalError('disk failure')):
            with self.assertRaises(sqlite3.OperationalError):
                self.apply(op('replace', 0, 4, 'HX170'))
        self.assertEqual(self.store.canonical_text, 'H170')
        self.assertEqual(self.store.revision, 1)
        self.assertEqual(self.store.segments[0].revision_modified, 1)
        self.assertEqual(len(self.store.history()), 1)

    def test_nan_raw_timestamp_is_preserved_without_failing_append(self):
        evidence = ASREvidence('nan', 0, 1, '原文', raw_result=[{'timestamp': [[0, float('nan')]]}])
        self.store.append(StabilityBuffer().push(evidence))
        self.assertEqual(self.store.canonical_text, '原文')
        self.assertEqual(self.store.evidence('nan')['raw_result'][0]['timestamp'][0][1],
                         {'nonfinite_raw_value': 'nan'})

    def test_dataclass_patch_cannot_bypass_parser(self):
        add(self.store, '原文')
        snapshot = self.store.snapshot()
        self.engine.apply(Patch.parse(patch_for(snapshot, op('insert', 2, 2, '。'))), snapshot)
        self.assertEqual(self.store.canonical_text, '原文。')

    def test_persistence_export_and_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'meeting.sqlite3'
            store = TranscriptStore(path)
            add(store, 'x' * 1030)
            snapshot = store.snapshot()
            RevisionEngine(store).apply(patch_for(snapshot, op('delete', 1000, 1030)), snapshot)
            store.close()
            reopened = TranscriptStore(path)
            try:
                self.assertEqual(reopened.frozen_boundary, 6)
                self.assertEqual(reopened.revision, 2)
                self.assertEqual(reopened.segments[0].raw_asr_text, 'x' * 1030)
                self.assertEqual(len(reopened.history()), 2)
                reopened.export_txt(Path(folder) / 'out.txt')
                self.assertEqual((Path(folder) / 'out.txt').read_text(), 'x' * 1000)
            finally:
                reopened.close()

    def test_segment_edits_match_independent_flat_string_reference(self):
        rng = random.Random(812)
        for trial in range(60):
            store = TranscriptStore()
            try:
                for i in range(5):
                    add(store, ''.join(rng.choice('中文ab😀') for _ in range(rng.randrange(0, 9))), str(i))
                for _ in range(10):
                    text = store.canonical_text
                    start = rng.randrange(len(text) + 1)
                    end = rng.randrange(start, len(text) + 1)
                    replacement = rng.choice(['', 'HX170', '中😀', 'x'])
                    if start == end and not replacement:
                        continue
                    kind = 'insert' if start == end else ('delete' if not replacement else 'replace')
                    snapshot = store.snapshot()
                    RevisionEngine(store).apply(patch_for(snapshot, op(kind, start, end, replacement)), snapshot)
                    self.assertEqual(store.canonical_text, text[:start] + replacement + text[end:])
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
