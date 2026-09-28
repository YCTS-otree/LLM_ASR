"""Render the unchanged Qt layout with a synthetic, patched transcript for QA."""
import os
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from unittest.mock import Mock, patch
from app import Window
from evidence import ASREvidence
from llm_pipeline import MeetingPipeline
from transcript_store import TranscriptStore
from engines import ROOT


def main():
    application = QApplication([])
    # Offscreen Qt does not discover native system fonts on this Windows runtime.
    # Load an existing system font for QA; no font installation/product override.
    font_file = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'msyh.ttc'
    if font_file.exists():
        identifier = QFontDatabase.addApplicationFont(str(font_file))
        families = QFontDatabase.applicationFontFamilies(identifier)
        if families:
            application.setFont(QFont(families[0], 10))
    with patch('app.input_devices', return_value=([(1, 'Default microphone', 'MME')], 1)):
        window = Window()
    store = TranscriptStore()
    pipeline = MeetingPipeline(store)
    pipeline.accept(ASREvidence('demo', 0, 2, '我们我们讨论H170与PCIE接口，，继续检查连接。'))
    pipeline.close()
    window.session = Mock(store=store)
    window.session.isRunning.return_value = False
    window.refresh_transcript()
    assert window.output.toPlainText() == '我们讨论HX170与PCIe接口，继续检查连接。'
    window.state.setText('已停止 · SQLite 与 TXT 已保存 · 未实时转写 0 段 · 设备溢出 0 次 · 模型保持常驻')
    window.path_label.setText('转录数据库：' + str(ROOT / 'transcripts' / '20260928_000000_paraformer_large_fp16.sqlite3'))
    (ROOT / 'logs').mkdir(exist_ok=True)
    window.show()
    for width, height in [(900, 700), (720, 600)]:
        window.resize(width, height)
        application.processEvents()
        assert window.width() == width, (window.width(), width)
        assert window.height() == height, (window.height(), height)
        assert window.output.height() > 40
        assert window.state.geometry().bottom() < window.output.geometry().top()
        window.grab().save(str(ROOT / 'logs' / f'ui_{width}.png'))
    window.session = None
    window.correction_model.setCurrentIndex(window.correction_model.findData('both'))
    window.apply_correction_selection()
    window.writable_window.setValue(2048)
    window.folder.setText('transcripts')
    window.path_label.setText('对比结果：各自保存 TXT 与 SQLite，原始 ASR 与修改历史保留。')
    window.llm_state.setText('界面布局测试 · 未加载模型')
    store.record_event('LLM_REQUEST',dict(result='APPLIED',latency=12.34,partial_failures=2))
    window.session=Mock(store=store,comparison_store=store)
    window.session.isRunning.return_value=False
    window.refresh_transcript()
    for width,height in [(900,700),(720,600)]:
        window.resize(width,height)
        application.processEvents()
        assert window.output.height()>40
        assert window.output.geometry().right()<window.comparison_output.geometry().left()
        window.grab().save(str(ROOT/'logs'/f'ui_dual_{width}.png'))
    window.session=None
    window.close()
    store.close()
    print('Patched transcript UI rendered at 900x700 and 720x600.')


if __name__ == '__main__':
    main()
