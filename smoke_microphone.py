"""Opt-in real microphone smoke test through the existing Qt Start/Stop flow.

Runs two short sessions and verifies that both use the same resident Engine.
Does not require speech; speech recognition quality needs a spoken test separately.
"""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--microphone', type=int)
    parser.add_argument('--sessions', type=int, default=2)
    parser.add_argument('--local-llm', action='store_true')
    parser.add_argument('--arm-file', type=Path, help='Wait for this file after models are ready')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--require-speech', action='store_true', help='Fail if ASR produces no spoken text')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error('--seconds must be between 1 and 30')
    if not 1 <= args.sessions <= 5:
        parser.error('--sessions must be between 1 and 5')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    import soundfile as sf
    from app import Window
    from engines import ROOT
    from observability import configure_logging
    configure_logging(ROOT / 'logs')
    qt = QApplication([])
    from PySide6.QtGui import QFontDatabase, QFont
    font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    if font_id >= 0:
        qt.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
    window = Window()
    if not args.local_llm:
        window.enable_correction.setChecked(False)
    window.folder.setText(str(ROOT / 'transcripts' / 'smoke'))
    if args.threshold is not None:
        window.threshold.setValue(args.threshold)
    if args.microphone is not None:
        index = window.microphone.findData(args.microphone)
        if index < 0:
            raise ValueError('Microphone endpoint not found')
        window.microphone.setCurrentIndex(index)
    results, errors = [], []
    state = {'first_engine': None, 'start_time': None, 'stopping': False, 'level_events': 0,
             'begun': False, 'announced': False, 'peak_rms': 0.0}
    window.show()

    def failed(message):
        errors.append(message)

    # Avoid a modal dialog in unattended verification; failures remain fatal.
    window.failed = failed

    def begin():
        state.update(start_time=None, stopping=False, level_events=0, begun=True, peak_rms=0.0)
        print('RECORDING', flush=True)
        window.start.click()
        if window.session is None:
            failed('UI did not create a session')
            qt.quit()
            return
        window.session.level.connect(lambda rms: state.update(level_events=state['level_events'] + 1,
                                                              peak_rms=max(state['peak_rms'], rms)))
        window.session.finished.connect(completed)

    def completed():
        session = window.session
        recorder = session.recorder
        try:
            if recorder is None or recorder.error:
                raise RuntimeError(recorder.error if recorder else 'Capture never started')
            if state['first_engine'] is None:
                state['first_engine'] = window.host.engine
            if window.host.engine is not state['first_engine']:
                raise AssertionError('Model reloaded between sessions')
            info = sf.info(recorder.wav_path)
            assert info.frames == recorder.captured_samples and info.frames > 0
            assert state['level_events'] > 0
            assert window.output.toPlainText() == session.store.canonical_text
            assert window.start.isEnabled() and not window.stop.isEnabled()
            export = Path(session.store.path).with_suffix('.txt')
            assert export.read_text(encoding='utf-8') == session.store.canonical_text
            speech_segments=sum(bool(s.raw_asr_text.strip()) for s in session.store.segments)
            llm_requests=sum(e['type']=='LLM_REQUEST' for e in session.store.events())
            results.append(dict(samples=info.frames, rate=info.samplerate, wav_format=info.format,
                                segments=len(session.store.segments), revisions=session.store.revision,
                                speech_segments=speech_segments, llm_requests=llm_requests,
                                microphone=window.microphone.currentText(), threshold=window.threshold.value(),
                                dropped=recorder.dropped, device_overflows=recorder.overflows,
                                level_events=state['level_events'], model_reused=True,
                                peak_rms=state['peak_rms'], llm_load_count=window.local_backend.load_count,
                                database=str(session.store.path), wav=recorder.wav_path))
            if args.require_speech and not speech_segments:
                failed('Capture worked but ASR produced no speech; check microphone and threshold')
            if args.require_speech and args.local_llm and not llm_requests:
                failed('No real ASR-to-Qwen request was observed')
        except Exception as exc:
            failed(str(exc))
        if len(results) < args.sessions and not errors:
            QTimer.singleShot(50, begin)
        else:
            window.grab().save(str(ROOT / 'logs' / 'smoke_ui.png'))
            window.close()
            qt.quit()

    def poll():
        if not state['begun'] and window.host.engine is not None and not window.loader.isRunning():
            if args.local_llm and not window.local_backend.ready:
                return
            if not state['announced']:
                print('READY_FOR_SPEECH', flush=True)
                state['announced'] = True
            if not args.arm_file or args.arm_file.is_file():
                begin()
        session = window.session
        if not session or not session.isRunning():
            return
        if session.recorder and session.recorder.captured_samples > 0 and state['start_time'] is None:
            state['start_time'] = time.monotonic()
        if state['start_time'] is not None and time.monotonic() - state['start_time'] >= args.seconds and not state['stopping']:
            state['stopping'] = True
            window.stop.click()

    def timeout():
        failed('Microphone smoke test timed out')
        if window.session and window.session.isRunning():
            window.session.request_stop('shutdown_flush')
        else:
            window.close()

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(50)
    QTimer.singleShot(180000, timeout)
    window.prepare_model()
    window.loader.error.connect(failed)
    qt.exec()
    summary = dict(passed=not errors and len(results) == args.sessions, sessions=results, errors=errors)
    (ROOT / 'logs' / 'microphone_smoke.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
