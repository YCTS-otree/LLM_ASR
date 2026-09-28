import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import unittest
from unittest.mock import Mock, patch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from app import Window
from transcript_store import TranscriptStore
from evidence import ASREvidence, StabilityBuffer
from patches import RevisionEngine
from test_transcript import patch_for, op


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.items = [(1, 'Internal', 'MME'), (2, 'RODE', 'WASAPI')]
        self.scanner = patch('app.input_devices', return_value=(self.items, 1))
        self.scan = self.scanner.start()
        self.window = Window()

    def tearDown(self):
        self.window.session = None
        self.window.close()
        self.scanner.stop()

    def test_refresh_retains_selected_name_across_reindex(self):
        self.window.microphone.setCurrentIndex(1)
        self.scan.return_value = ([(3, 'RODE', 'WASAPI'), (7, 'Internal', 'MME')], 7)
        self.window.refresh.click()
        self.scan.assert_called_with(True)
        self.assertEqual(self.window.microphone.currentData(), 3)
        self.assertEqual(tuple(self.window.microphone.currentData(Qt.UserRole + 1)), ('RODE', 'WASAPI'))

    def test_refresh_prohibited_during_session(self):
        self.window.session = Mock()
        self.window.session.isRunning.return_value = True
        self.scan.reset_mock()
        self.window.load_microphones(True)
        self.scan.assert_not_called()
        self.window.set_busy(True)
        self.assertFalse(self.window.refresh.isEnabled())

    def test_precision_changes_device(self):
        self.window.precision.setCurrentIndex(self.window.precision.findData('int8'))
        self.assertEqual(self.window.device.currentData(), 'cpu')
        self.window.precision.setCurrentIndex(self.window.precision.findData('fp16'))
        self.assertEqual(self.window.device.currentData(), 'auto')

    def test_failed_refresh_discards_stale_indices(self):
        self.scan.side_effect = RuntimeError('device error')
        self.window.refresh.click()
        self.assertEqual(self.window.microphone.count(), 0)
        self.assertIn('device error', self.window.state.text())

    def test_ui_reads_latest_store_after_patch_not_signal_payload(self):
        store = TranscriptStore()
        try:
            store.append(StabilityBuffer().push(ASREvidence('s', 0, 1, 'H170')))
            self.window.session = Mock(store=store)
            self.window.refresh_transcript('old payload')
            self.assertEqual(self.window.output.toPlainText(), 'H170')
            snapshot = store.snapshot()
            RevisionEngine(store).apply(patch_for(snapshot, op('replace', 0, 4, 'HX170')), snapshot)
            self.window.refresh_transcript('H170')
            self.assertEqual(self.window.output.toPlainText(), 'HX170')
        finally:
            self.window.session = None
            store.close()

    def test_close_requests_shutdown_flush(self):
        session = Mock()
        session.isRunning.return_value = True
        self.window.session = session
        event = Mock()
        self.window.closeEvent(event)
        session.request_stop.assert_called_once_with('shutdown_flush')
        event.ignore.assert_called_once()

    def test_invalid_llm_configuration_keeps_asr_controls_available(self):
        with patch.dict(os.environ, {'LOCAL_LLM_DTYPE':'unsupported'}):
            window=Window()
        try:
            self.assertFalse(window.enable_correction.isChecked())
            self.assertIn('Error',window.llm_state.text())
            self.assertTrue(window.start.isEnabled())
            self.assertTrue(window.microphone.isEnabled())
        finally:
            window.close()

    def test_file_import_does_not_require_microphone(self):
        self.window.microphone.clear()
        self.window.enable_correction.setChecked(False)
        self.window.host.engine=Mock()
        self.window.host.options=(self.window.device.currentData(),self.window.precision.currentData(),self.window.config)
        self.window.writable_window.setValue(2048)
        with patch('app.QFileDialog.getOpenFileName',return_value=('C:/recording.wav','')), patch('app.Session') as session:
            self.window.import_button.click()
            self.assertEqual(session.call_args.args[-1],'C:/recording.wav')
            self.assertEqual(session.call_args.kwargs['window_chars'],2048)
            session.return_value.start.assert_called_once()
            self.assertFalse(self.window.import_button.isEnabled())
            self.assertFalse(self.window.writable_window.isEnabled())

    def test_model_selector_supports_single_small_and_dual_independent_members(self):
        self.window.correction_model.setCurrentIndex(self.window.correction_model.findData('0.8b'))
        self.assertEqual(self.window.local_backend.config.model_size,'2b')
        self.window.apply_correction_selection()
        self.assertEqual(self.window.local_backend.config.model_size,'0.8b')
        self.assertIn('0.8B',self.window.output_title.text())
        self.window.correction_model.setCurrentIndex(self.window.correction_model.findData('both'))
        self.window.apply_correction_selection()
        self.assertEqual(len(self.window.local_backend.members),2)
        self.assertEqual(self.window.llm_precision.currentData(),'int4')
        self.assertEqual(self.window.local_backend.members[1].config.dtype,'bf16')
        self.assertFalse(self.window.comparison_output.isHidden())
        self.window.set_busy(True)
        self.assertFalse(self.window.correction_model.isEnabled())
        self.assertFalse(self.window.llm_precision.isEnabled())
        self.assertFalse(self.window.comparison_precision.isEnabled())

    def test_parameter_changes_and_enable_never_start_loading(self):
        original=self.window.local_backend
        with patch('app.ModelLoader') as asr,patch('app.LocalModelLoader') as llm:
            self.window.correction_model.setCurrentIndex(2)
            self.window.llm_precision.setCurrentIndex(2)
            self.window.comparison_precision.setCurrentIndex(1)
            self.window.enable_correction.setChecked(False)
            self.window.enable_correction.setChecked(True)
            self.assertIs(self.window.local_backend,original)
            asr.assert_not_called();llm.assert_not_called()
            self.window.load_button.click()
            asr.return_value.start.assert_called_once()
            self.assertEqual(self.window.local_backend.mode,'both')
            self.assertEqual(self.window.local_backend.config.dtype,'fp16')
            self.assertFalse(self.window.load_button.isEnabled())
        self.window.loader=None

    def test_begin_with_unloaded_or_changed_models_does_not_load_or_capture(self):
        with patch('app.Session') as session,patch('app.ModelLoader') as loader:
            self.window.begin('example.wav')
            session.assert_not_called();loader.assert_not_called()
            self.assertIn('Load model',self.window.state.text())

    def test_layout_at_default_and_smaller_window(self):
        self.window.show()
        for width, height in [(900, 700), (720, 600)]:
            self.window.resize(width, height)
            self.app.processEvents()
            self.assertGreater(self.window.output.height(), 40)
            self.assertLess(self.window.state.geometry().bottom(), self.window.output.geometry().top())
            self.assertLess(self.window.output.geometry().bottom(), self.window.path_label.geometry().top())


if __name__ == '__main__':
    unittest.main()
