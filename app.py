import sys
import queue
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import replace
from PySide6.QtCore import QThread, Signal, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLabel, QComboBox, QPushButton, QPlainTextEdit,
    QLineEdit, QFileDialog, QProgressBar, QMessageBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QDialog, QDialogButtonBox)
from engines import ROOT
from audio_pipeline import Capture
from file_capture import FileCapture
from devices import input_devices, select_device
from model_host import ModelHost
from settings import ASRConfig, LocalLLMConfig
from correction_models import CorrectionModels
from qwen_spec import MODEL_SPECS
from transcript_store import TranscriptStore
from llm_pipeline import MeetingPipeline
from observability import configure_logging
from version import __version__


class ModelLoader(QThread):
    ready = Signal(str)
    error = Signal(str)

    def __init__(self, host, device, precision, config):
        super().__init__()
        self.host, self.device, self.precision, self.config = host, device, precision, config

    def run(self):
        try:
            engine = self.host.load(self.device, self.precision, self.config)
            self.ready.emit('模型常驻就绪 · ' + engine.description)
        except Exception as exc:
            self.error.emit(str(exc))


class LocalModelLoader(QThread):
    error = Signal(str)

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def run(self):
        try:
            self.backend.load()
        except Exception as exc:
            self.error.emit(str(exc))


class Session(QThread):
    status = Signal(str)
    text = Signal(str)
    level = Signal(float)
    saved = Signal(str)
    error = Signal(str)

    def __init__(self, device, precision, microphone, folder, threshold, host, config, backend=None, input_path=None, window_chars=1024, punctuation_mode='combined'):
        super().__init__()
        self.options = device, precision, microphone, folder, threshold
        self.stop_event = threading.Event()
        self.host, self.config = host, config
        self.recorder = None
        self.stop_reason = 'manual_stop'
        self.backend = backend
        self.input_path = input_path
        self.window_chars = window_chars
        from pause_evidence import PUNCTUATION_MODES
        if punctuation_mode not in PUNCTUATION_MODES:raise ValueError('Invalid punctuation mode')
        self.punctuation_mode=punctuation_mode

    def request_stop(self, reason='manual_stop'):
        self.stop_reason = reason
        if self.recorder:
            self.recorder.end_reason = reason
        self.stop_event.set()

    def run(self):
        recorder = store = None
        stores, pipelines = [], []
        self.comparison_store = None
        failed_segments = 0
        exported = False
        logger = configure_logging(ROOT / 'logs')
        try:
            device, precision, microphone, folder, threshold = self.options
            self.status.emit('正在准备常驻模型，请稍候…')
            engine = self.host.load(device, precision, self.config)
            if self.input_path and self.backend and self.backend.enabled and not self.backend.ready:
                self.status.emit('正在准备离线纠错模型…')
                try:
                    self.backend.load()
                except Exception:
                    logger.exception('Local correction unavailable; importing with ASR only')
            if self.stop_event.is_set():
                return
            Path(folder).mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            stem = f'{stamp}_{"import_" if self.input_path else ""}paraformer_large_{precision}'
            backends = getattr(self.backend, 'members', (self.backend,))
            size = getattr(getattr(backends[0], 'config', None), 'model_size', '2b')
            suffix = '_2b' if len(backends)==2 else ('_0_8b' if size=='0.8b' else '')
            database = Path(folder) / (stem + suffix + '.sqlite3')
            store = TranscriptStore(database, window_chars=self.window_chars)
            stores.append(store)
            self.store = store
            if len(backends)==2:
                self.comparison_store = TranscriptStore(Path(folder) / (stem + '_0_8b.sqlite3'), window_chars=self.window_chars)
                stores.append(self.comparison_store)
            for branch_store, branch_backend in zip(stores, backends):
                branch_store.record_event('SESSION_START', {'version': __version__, 'device': engine.device,
                                                'precision': precision, 'beam_size': self.config.beam_size,
                                                'nbest': self.config.nbest,
                                                'window_chars': self.window_chars,
                                                'punctuation_mode': self.punctuation_mode,
                                                'backend': getattr(branch_backend, 'source', 'MOCK_LLM'),
                                                'comparison_id': stem if len(backends)==2 else None,
                                                'correction_model': getattr(branch_backend, 'model_id', None),
                                                'mode': 'file' if self.input_path else 'microphone',
                                                'source_file': str(self.input_path) if self.input_path else None})
            self.saved.emit(' | '.join(str(s.path) for s in stores))
            def changed():
                # The signal is a notification. The UI reads the store's latest
                # state, so cross-thread delivery cannot repaint an old snapshot.
                self.text.emit('')
            for branch_store, branch_backend in zip(stores,backends):
                pipelines.append(MeetingPipeline(branch_store, changed, branch_backend,punctuation_mode=self.punctuation_mode))
            def record_event(kind, data):
                for branch_store in stores:
                    branch_store.record_event(kind, data)
            wav_path = ROOT / 'recordings' / (stem + '.wav')
            recorder_class = FileCapture if self.input_path else Capture
            recorder = recorder_class(self.input_path if self.input_path else microphone, str(wav_path), self.stop_event, threshold,
                               self.level.emit, lambda s: self.status.emit(s + ' · ' + engine.description),
                               event=record_event)
            self.recorder = recorder
            recorder.end_reason = 'end_of_file' if self.input_path else self.stop_reason
            recorder.start()
            self.status.emit('模型已加载 · ' + engine.description)
            while not recorder.done.is_set() or not recorder.segments.empty():
                try:
                    chunk = recorder.segments.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    evidence = engine.transcribe(chunk)
                    from pause_evidence import energy_pauses
                    evidence=replace(evidence,low_energy_spans=energy_pauses(chunk.audio,chunk.audio_rate,threshold),pause_threshold=threshold)
                except Exception as exc:
                    failed_segments += 1
                    logger.exception('ASR failed segment=%s', chunk.segment_id)
                    record_event('ASR_FAILED', {'segment_id': chunk.segment_id,
                        'start_sample': chunk.start_sample, 'end_sample': chunk.end_sample,
                        'sample_rate': chunk.sample_rate, 'wav_path': str(wav_path), 'error': str(exc)})
                    self.status.emit('某段识别失败，已记录音频位置，继续处理')
                    continue
                for branch in pipelines:
                    branch.accept(evidence)
                if evidence.fallback_reason:
                    record_event('ASR_PRECISION_FALLBACK',dict(segment_id=evidence.segment_id,
                        reason=evidence.fallback_reason,actual_precision=evidence.inference_precision))
                if evidence.punctuation_error:
                    record_event('PUNCTUATION_FAILED',dict(segment_id=evidence.segment_id,error=evidence.punctuation_error))
                if self.input_path:
                    # Offline throughput may wait; microphone ASR never waits
                    # for LLM. Avoid freezing unprocessed text during fast import.
                    for branch in pipelines:
                        branch.worker.requests.join()
            if recorder.error:
                raise RuntimeError(recorder.error)
            for branch in pipelines:
                branch.close()
            pipelines = []
            for branch_store in stores:
                branch_store.export_txt(Path(branch_store.path).with_suffix('.txt'))
            exported = True
            completion = '离线转录完成' if self.input_path and not self.stop_event.is_set() else '已停止'
            self.status.emit(f'{completion} · SQLite 与 TXT 已保存 · 未实时转写 {recorder.dropped} 段'
                             f' · 识别失败 {failed_segments} 段 · 设备溢出 {recorder.overflows} 次 · 模型保持常驻')
        except Exception as exc:
            logger.exception('session failed')
            self.error.emit(str(exc))
        finally:
            self.stop_event.set()
            if recorder:
                if hasattr(recorder, 'abort'):
                    recorder.abort.set()
                recorder.join()
            for branch in pipelines:
                branch.close()
            # UI may still have queued change notifications. Keep the store open
            # until the UI retires this Session, after finished has been delivered.
            if not exported:
                for branch_store in stores:
                    try:
                        branch_store.export_txt(Path(branch_store.path).with_suffix('.txt'))
                    except Exception:
                        logger.exception('TXT export failed; SQLite remains authoritative')

    def retire(self):
        if getattr(self, 'comparison_store', None) is not None:
            self.comparison_store.close()
            self.comparison_store = None
        if hasattr(self, 'store'):
            self.store.close()
            del self.store


class Window(QMainWindow):
    llm_state_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.session = None
        self.host = ModelHost()
        self.config = ASRConfig.from_environment()
        self.loader = None
        self.llm_loader = None
        self.llm_config_error = None
        try:
            llm_config = LocalLLMConfig.from_environment()
        except (TypeError, ValueError) as exc:
            # An optional enhancement's configuration cannot disable capture/ASR.
            self.llm_config_error = str(exc)
            llm_config = LocalLLMConfig()
        self.base_llm_config = llm_config
        from deepseek_backend import DeepSeekConfig
        self.deepseek_config = DeepSeekConfig()
        self.local_backend = CorrectionModels(llm_config, self.llm_state_changed.emit)
        self.closing = False
        self.setWindowTitle(f'本地语音转写 · FunASR Paraformer-large · v{__version__}')
        self.glossary_action=QPushButton('术语库')
        self.glossary_action.clicked.connect(self.edit_glossary)
        self.resize(900, 700)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(4)
        title = QLabel('本地语音转写')
        title.setStyleSheet('font-size: 26px; font-weight: 700;')
        layout.addWidget(title)
        layout.addWidget(QLabel('录音 → Paraformer + 基础标点 → 上下文校对 → 版本化转录'))
        form = QFormLayout()
        form.setVerticalSpacing(1)
        self.model = QComboBox()
        self.model.addItem('达摩 Paraformer-large · 220M · 中文（ModelScope 国内源）')
        form.addRow('识别模型', self.model)
        self.device = QComboBox()
        for label, value in [('自动（优先 NVIDIA GPU）', 'auto'), ('CPU', 'cpu'), ('NVIDIA GPU', 'cuda')]:
            self.device.addItem(label, value)
        form.addRow('计算设备', self.device)
        self.precision = QComboBox()
        for label, value in [('FP16 · GPU 混合精度', 'fp16'), ('BF16 · GPU 混合精度', 'bf16'),
                             ('INT8 · CPU 动态量化', 'int8'), ('FP32 · 标准精度', 'fp32')]:
            self.precision.addItem(label, value)
        self.precision.currentIndexChanged.connect(self.precision_changed)
        form.addRow('计算精度', self.precision)
        self.microphone = QComboBox()
        microw = QHBoxLayout()
        microw.addWidget(self.microphone, 1)
        self.refresh = QPushButton('刷新')
        self.refresh.clicked.connect(lambda: self.load_microphones(refresh=True))
        microw.addWidget(self.refresh)
        form.addRow('麦克风', microw)
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(3)
        self.threshold.setRange(.001, .100)
        self.threshold.setSingleStep(.001)
        self.threshold.setValue(.008)
        self.threshold.setToolTip('声音较轻时调低；环境噪声较大时调高。')
        self.punctuation_mode=QComboBox()
        for label,value in [('基础标点 + 停顿','combined'),('仅基础标点','baseline'),('仅停顿证据','pauses')]:
            self.punctuation_mode.addItem(label,value)
        self.punctuation_mode.setToolTip('新会话生效，无需重载模型。基础标点是文本模型预测；停顿是低能量估计。仅停顿模式不使用基础标点初始化文本，但仍保留其证据。')
        punctuation_row=QHBoxLayout()
        punctuation_row.addWidget(self.threshold)
        punctuation_row.addWidget(QLabel('标点策略'))
        punctuation_row.addWidget(self.punctuation_mode,1)
        form.addRow('声音触发阈值', punctuation_row)
        self.folder = QLineEdit(str(ROOT / 'transcripts'))
        outrow = QHBoxLayout()
        outrow.addWidget(self.folder)
        self.browse = QPushButton('选择目录')
        self.browse.clicked.connect(self.choose_folder)
        outrow.addWidget(self.browse)
        form.addRow('转录保存目录', outrow)
        self.correction_model = QComboBox()
        for label, value in [('Qwen3.5-2B', '2b'), ('Qwen3.5-0.8B', '0.8b'), ('2B / 0.8B 双模型对比', 'both'), ('DeepSeek 在线 API','deepseek')]:
            self.correction_model.addItem(label, value)
        self.correction_model.setCurrentIndex(self.correction_model.findData(llm_config.model_size))
        self.llm_precision = QComboBox()
        for label, value in [('BF16', 'bf16'), ('INT4 (NF4)', 'int4'), ('FP16', 'fp16')]:
            self.llm_precision.addItem(label, value)
        self.llm_precision.setCurrentIndex(self.llm_precision.findData(llm_config.dtype))
        self.comparison_precision = QComboBox()
        for label,value in [('BF16','bf16'),('INT4','int4'),('FP16','fp16')]:
            self.comparison_precision.addItem(label,value)
        self.comparison_precision.setCurrentIndex(self.comparison_precision.findData('bf16'))
        self.comparison_precision_label = QLabel('0.8B')
        self.comparison_precision_label.hide()
        self.comparison_precision.hide()
        model_row = QHBoxLayout()
        model_row.addWidget(self.correction_model, 1)
        model_row.addWidget(QLabel('LLM 精度'))
        model_row.addWidget(self.llm_precision)
        model_row.addWidget(self.comparison_precision_label)
        model_row.addWidget(self.comparison_precision)
        form.addRow('纠错模型', model_row)
        self.thinking = QCheckBox('Qwen 思考')
        self.thinking.setToolTip('启用后允许模型先推理再校对，速度较慢；需点击 Load model 应用。')
        self.thinking.toggled.connect(self.models_pending)
        self.api_settings = QPushButton('LLM 请求设置')
        self.api_settings.clicked.connect(self.edit_api_settings)
        self.proofreading_options = QHBoxLayout()
        self.proofreading_options.addWidget(self.thinking)
        self.proofreading_options.addWidget(self.api_settings)
        self.proofreading_options.addWidget(self.glossary_action)
        self.writable_window = QSpinBox()
        self.writable_window.setRange(256, 4096)
        self.writable_window.setSingleStep(256)
        self.writable_window.setValue(2048)
        self.writable_window.setSuffix(' 字')
        self.writable_window.setToolTip('最近允许回改的 Unicode 字符数，可设为2048。新会话生效；已冻结历史不会解冻。扩大窗口不代表每次重写整段。')
        self.proofreading_options.insertWidget(0,self.writable_window)
        form.addRow('回改窗口 / 校对', self.proofreading_options)
        self.enable_correction = QCheckBox('启用上下文校对')
        self.enable_correction.setToolTip('实验功能：可能漏改或误改。原始识别和修改历史始终保留在会话数据库。')
        self.enable_correction.setChecked(True)
        self.enable_correction.toggled.connect(self.toggle_correction)
        self.llm_state = QLabel('未加载 · 请先配置参数')
        self.llm_state.setWordWrap(True)
        self.llm_state_changed.connect(self.llm_state.setText)
        local_row = QHBoxLayout()
        local_row.addWidget(self.enable_correction)
        local_row.addWidget(self.llm_state, 1)
        self.load_button = QPushButton('Load model')
        self.load_button.clicked.connect(self.prepare_model)
        local_row.addWidget(self.load_button)
        self.logs_button = QPushButton('日志')
        self.logs_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT / 'logs'))))
        local_row.addWidget(self.logs_button)
        form.addRow('LLM', local_row)
        layout.addLayout(form)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setFormat('输入音量 %p%')
        layout.addWidget(self.meter)
        row = QHBoxLayout()
        self.start = QPushButton('开始录音')
        self.start.clicked.connect(lambda: self.begin())
        self.import_button = QPushButton('导入录音')
        self.import_button.clicked.connect(self.import_audio)
        self.stop = QPushButton('停止并保存')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.end)
        self.open_folder = QPushButton('打开保存目录')
        self.open_folder.clicked.connect(self.show_folder)
        row.addWidget(self.start)
        row.addWidget(self.import_button)
        row.addWidget(self.stop)
        row.addWidget(self.open_folder)
        layout.addLayout(row)
        self.state = QLabel('先调整参数，再点击 Load model；加载完成后开始录音或导入。')
        self.state.setWordWrap(True)
        layout.addWidget(self.state)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText('当前转录会随已验证 Patch 更新，原始 ASR 和修订历史保存在 SQLite。\n停止时导出 TXT；音频保存到 recordings。')
        self.output_title = QLabel(f'Qwen3.5-{llm_config.model_size.upper()}')
        self.comparison_title = QLabel('Qwen3.5-0.8B')
        self.comparison_output = QPlainTextEdit()
        self.comparison_output.setReadOnly(True)
        self.comparison_output.setPlaceholderText('相同 ASR 的独立纠错结果；不读取左侧模型的修改。')
        self.comparison_title.hide()
        self.comparison_output.hide()
        titles = QHBoxLayout()
        titles.addWidget(self.output_title, 1)
        titles.addWidget(self.comparison_title, 1)
        layout.addLayout(titles)
        outputs = QHBoxLayout()
        outputs.addWidget(self.output, 1)
        outputs.addWidget(self.comparison_output, 1)
        layout.addLayout(outputs, 1)
        self.path_label = QLabel('每次录音生成独立文件，停止后会处理剩余语音。')
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        self.setStyleSheet('QWidget { font-size: 14px; } QLineEdit, QComboBox, QPushButton { padding: 5px; } QPlainTextEdit { font-size: 17px; }')
        self.load_microphones()
        self.correction_model.currentIndexChanged.connect(self.change_correction_models)
        self.llm_precision.currentIndexChanged.connect(self.change_correction_models)
        self.comparison_precision.currentIndexChanged.connect(self.change_correction_models)
        self.device.currentIndexChanged.connect(self.models_pending)
        self.precision.currentIndexChanged.connect(self.models_pending)
        self._selection_mode = self.correction_model.currentData()
        if self.llm_config_error:
            self.enable_correction.setChecked(False)
            self.enable_correction.setEnabled(False)
            self.llm_state.setText('Qwen3.5-2B · Error · 配置无效：' + self.llm_config_error)

    def prepare_model(self):
        if ((self.session and self.session.isRunning()) or
                (self.loader and self.loader.isRunning()) or
                (self.llm_loader and self.llm_loader.isRunning())):
            return
        self.apply_correction_selection()
        self.set_busy(True)
        self.stop.setEnabled(False)
        self.state.setText('正在后台加载常驻模型…')
        self.loader = ModelLoader(self.host, self.device.currentData(), self.precision.currentData(), self.config)
        self.loader.ready.connect(self.state.setText)
        self.loader.error.connect(lambda message: self.state.setText('模型加载失败：' + message))
        self.loader.finished.connect(self.model_loaded)
        self.loader.start()

    def model_loaded(self):
        self.set_busy(False)
        if self.closing:
            self.close()
        elif self.host.engine is not None and self.enable_correction.isChecked():
            self.prepare_local_model()

    def prepare_local_model(self):
        if (self.local_backend.ready and self.local_backend.mode!='deepseek') or (self.llm_loader and self.llm_loader.isRunning()):
            return
        self.llm_loader = LocalModelLoader(self.local_backend)
        self.set_busy(True)
        self.stop.setEnabled(False)
        self.llm_loader.error.connect(lambda message: self.llm_state.setText('Error · ' + message))
        self.llm_loader.finished.connect(self.local_models_loaded)
        self.llm_loader.start()

    def local_models_loaded(self):
        busy = bool(self.session and self.session.isRunning())
        self.set_busy(busy)
        if self.closing:
            self.close()

    def change_correction_models(self, _index=None):
        if (self.session and self.session.isRunning()) or (self.llm_loader and self.llm_loader.isRunning()):
            return
        mode = self.correction_model.currentData()
        if mode == 'both' and self._selection_mode != 'both':
            self.llm_precision.blockSignals(True)
            self.llm_precision.setCurrentIndex(self.llm_precision.findData('int4'))
            self.llm_precision.blockSignals(False)
        self._selection_mode = mode
        self.comparison_precision_label.setVisible(mode == 'both')
        self.comparison_precision.setVisible(mode == 'both')
        self.thinking.setEnabled(mode != 'deepseek')
        self.llm_precision.setEnabled(mode != 'deepseek')
        self.models_pending()

    def models_pending(self, _value=None):
        self.state.setText('参数已选择 · 点击 Load model 应用；不会自动加载。')

    def edit_api_settings(self):
        online=self.correction_model.currentData()=='deepseek'
        config=self.deepseek_config if online else self.base_llm_config
        dialog=QDialog(self)
        dialog.setWindowTitle('DeepSeek 请求设置' if online else 'Qwen 请求设置')
        layout=QFormLayout(dialog)
        note=QLabel('每次只修改目标片段，同时提供回改窗口中的前后文。目标越长，请求越少，但单次生成更长。\n保存后点击 Load model 应用。回改窗口在主界面设置，新会话生效。'+('\nDeepSeek 会接收文字、候选、匹配术语和停顿信息，不接收音频；按服务商规则计费。密钥留空则从运行目录 DEEPSEEK.key 读取。' if online else '\n本地小模型建议从128字开始；较长目标可能增加漏改和格式错误。'))
        note.setWordWrap(True)
        note.setMaximumWidth(560)
        layout.addRow(note)
        def integer(value,minimum,maximum,suffix=''):
            widget=QSpinBox();widget.setRange(minimum,maximum);widget.setValue(int(value));widget.setSuffix(suffix)
            return widget
        target=integer(config.target_chars,64,2048,' 字')
        tokens=integer(config.output_token_limit if online else config.max_new_tokens,256,32768 if online else 8192)
        batch=integer(config.timeout_seconds,30,1800,' 秒')
        layout.addRow('单次校对目标上限',target)
        layout.addRow('输出 token 上限',tokens)
        layout.addRow('整轮校对超时',batch)
        retry=QCheckBox('无修改时追加一次标点校对（增加请求）')
        retry.setChecked(config.retry_unchanged)
        layout.addRow(retry)
        inputs=integer(config.max_input_tokens,512,16384)
        if not online:layout.addRow('输入 token 上限',inputs)
        model=QComboBox()
        model.setEditable(True)
        model.addItems(['deepseek-flash','deepseek-v4-pro'])
        model.setCurrentText(self.deepseek_config.api_model)
        key=QLineEdit(self.deepseek_config.api_key)
        key.setEchoMode(QLineEdit.Password)
        key.setPlaceholderText('留空读取 DEEPSEEK.key')
        think=QCheckBox('启用思考')
        think.setChecked(self.deepseek_config.thinking)
        effort=QComboBox();effort.addItems(['low','high','max']);effort.setCurrentText(self.deepseek_config.reasoning_effort)
        timeout=integer(self.deepseek_config.request_timeout,5,600,' 秒')
        temperature=QDoubleSpinBox();temperature.setRange(0,2);temperature.setSingleStep(.1);temperature.setValue(self.deepseek_config.temperature)
        effort.setEnabled(think.isChecked());temperature.setEnabled(not think.isChecked())
        think.toggled.connect(effort.setEnabled);think.toggled.connect(lambda enabled:temperature.setEnabled(not enabled))
        if online:
            layout.addRow('模型 ID',model)
            layout.addRow('API Key',key)
            layout.addRow(think)
            layout.addRow('推理深度',effort)
            layout.addRow('温度（仅关闭思考时）',temperature)
            layout.addRow('单次请求超时',timeout)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec()==QDialog.Accepted:
            if not online:
                self.base_llm_config=replace(config,target_chars=target.value(),max_new_tokens=tokens.value(),max_input_tokens=inputs.value(),timeout_seconds=batch.value(),retry_unchanged=retry.isChecked())
                self.models_pending()
            elif model.currentText().strip():
                self.deepseek_config=replace(self.deepseek_config,api_model=model.currentText().strip(),api_key=key.text().strip(),thinking=think.isChecked(),
                    target_chars=target.value(),output_token_limit=tokens.value(),timeout_seconds=batch.value(),
                    retry_unchanged=retry.isChecked(),
                    reasoning_effort=effort.currentText(),request_timeout=timeout.value(),temperature=temperature.value())
                self.models_pending()
            else:
                self.state.setText('DeepSeek 模型 ID 不能为空，未修改配置。')
        key.clear()

    def edit_glossary(self):
        import json
        from glossary import load_terms,validate_terms,DEFAULT_FILE
        dialog=QDialog(self)
        dialog.setWindowTitle('术语库 · glossary.json')
        dialog.resize(640,480)
        layout=QVBoxLayout(dialog)
        note=QLabel('保存到当前运行目录的 glossary.json，新会话生效，无需重载模型。\n匹配的词条作为参考随文本发送给所选LLM；不直接替换原文。')
        note.setWordWrap(True);layout.addWidget(note)
        editor=QPlainTextEdit();layout.addWidget(editor,1)
        status=QLabel();status.setWordWrap(True);layout.addWidget(status)
        try:editor.setPlainText(json.dumps(load_terms(),ensure_ascii=False,indent=2))
        except (OSError,ValueError):
            editor.setPlainText('[]');status.setText('现有术语库不可读；仅点击保存才会覆盖。')
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)
        def save():
            try:
                data=validate_terms(json.loads(editor.toPlainText()))
                temporary=DEFAULT_FILE.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                temporary.replace(DEFAULT_FILE)
            except (OSError,ValueError):
                status.setText('保存失败：请检查JSON、term/aliases/note字段及目录写权限。')
                return
            self.state.setText('术语库已保存 · 下次会话生效，无需重载模型。')
            dialog.accept()
        buttons.accepted.connect(save)
        dialog.exec()

    def selected_correction_config(self):
        mode = self.correction_model.currentData()
        if mode == 'deepseek':
            return self.deepseek_config,mode,self.comparison_precision.currentData()
        size = '2b' if mode == 'both' else mode
        config = replace(self.base_llm_config, model_size=size, dtype=self.llm_precision.currentData(),
                         thinking=self.thinking.isChecked(),
                         model_path=self.base_llm_config.model_path if size == self.base_llm_config.model_size else str(MODEL_SPECS[size]['path']))
        return config, mode, self.comparison_precision.currentData()

    def correction_selection_matches(self):
        config, mode, comparison_dtype = self.selected_correction_config()
        return (self.local_backend.config == config and self.local_backend.mode == mode and
                (mode != 'both' or self.local_backend.members[1].config.dtype == comparison_dtype))

    def apply_correction_selection(self):
        if self.correction_selection_matches():
            return
        config, mode, comparison_dtype = self.selected_correction_config()
        size = config.model_size
        self.local_backend.close()
        self.local_backend = CorrectionModels(config, self.llm_state_changed.emit, mode, comparison_dtype)
        self.local_backend.set_enabled(self.enable_correction.isChecked())
        # Retire completed outputs before relabelling a different model.
        if self.session:
            self.session.retire()
            self.session = None
        self.output.clear()
        self.comparison_output.clear()
        self.output_title.setText(config.api_model if mode=='deepseek' else f'Qwen3.5-{size.upper()}')
        self.comparison_title.setVisible(mode == 'both')
        self.comparison_output.setVisible(mode == 'both')
        self.comparison_precision_label.setVisible(mode == 'both')
        self.comparison_precision.setVisible(mode == 'both')

    def toggle_correction(self, enabled):
        self.local_backend.set_enabled(enabled)
        if enabled and not self.local_backend.ready:
            self.llm_state.setText('待加载 · 点击 Load model')

    def refresh_transcript(self, _notification=''):
        if self.session and hasattr(self.session, 'store'):
            text = self.session.store.canonical_text
            if text != self.output.toPlainText():
                scroll = self.output.verticalScrollBar()
                at_end = scroll.value() >= scroll.maximum()
                position = scroll.value()
                self.output.setPlainText(text)
                scroll.setValue(scroll.maximum() if at_end else position)
            other = getattr(self.session, 'comparison_store', None)
            if isinstance(other, TranscriptStore):
                text = other.canonical_text
                if text != self.comparison_output.toPlainText():
                    scroll = self.comparison_output.verticalScrollBar()
                    at_end, position = scroll.value() >= scroll.maximum(), scroll.value()
                    self.comparison_output.setPlainText(text)
                    scroll.setValue(scroll.maximum() if at_end else position)
            for member, title, branch_store in zip(self.local_backend.members, (self.output_title, self.comparison_title), (self.session.store, other)):
                latest = branch_store.last_event('LLM_REQUEST') if isinstance(branch_store, TranscriptStore) else None
                latency = latest.get('latency') if latest else None
                outcome = '' if not latest else ('已应用' if latest['result']=='APPLIED' else ('无修改' if latest['result']=='NO_CHANGE' else '保留原文'))
                if latest and latest['result']=='APPLIED' and 'lexical_changed' in latest:
                    outcome='含字词修改' if latest['lexical_changed'] else '仅标点/格式修改'
                if latest and latest.get('partial_failures'):
                    outcome += f' · {latest["partial_failures"]}处未通过'
                if latest and latest.get('detail'):
                    outcome += ' · '+str(latest['detail'])[:80]
                title.setWordWrap(True)
                title.setText(getattr(member,'model_id',f'Qwen3.5-{member.config.model_size.upper()}') +
                              (f' · {latency:.2f}s · {outcome}' if latency is not None else ''))
                title.setToolTip('' if not latest else
                    f'本轮请求：{latest.get("request_count", "未知")}；输入 tokens：{latest.get("input_tokens", 0)}；输出 tokens：{latest.get("output_tokens", 0)}；推理 tokens：{latest.get("reasoning_tokens", 0)}\n'
                    f'包含数字单位规则规范化：{"是" if latest.get("written_number_normalization") else "否"}')

    def load_microphones(self, refresh=False):
        if self.session and self.session.isRunning():
            return
        previous = self.microphone.currentData(Qt.UserRole + 1)
        try:
            items, default = input_devices(refresh)
            self.microphone.clear()
            for index, name, host in items:
                self.microphone.addItem(f'{name} [{host}]', index)
                self.microphone.setItemData(self.microphone.count() - 1, (name, host), Qt.UserRole + 1)
            self.microphone.setCurrentIndex(select_device(items, tuple(previous) if previous else None, default))
            if not self.microphone.count():
                self.state.setText('未发现麦克风；可导入录音，或连接麦克风后刷新。')
            elif refresh:
                self.state.setText(f'已重新扫描音频设备 · {len(items)} 个输入端点（同一麦克风可能有多个接口）')
        except Exception as exc:
            self.microphone.clear()
            self.state.setText(f'无法读取麦克风：{exc}')

    def precision_changed(self):
        if self.precision.currentData() == 'int8':
            self.device.setCurrentIndex(self.device.findData('cpu'))
        elif self.precision.currentData() in ('fp16', 'bf16'):
            self.device.setCurrentIndex(self.device.findData('auto'))

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择 TXT 保存目录', self.folder.text())
        if folder:
            self.folder.setText(folder)

    def show_folder(self):
        path = Path(self.folder.text()).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        except OSError as exc:
            QMessageBox.warning(self, '目录不可用', str(exc))

    def set_busy(self, busy):
        for widget in (self.start, self.import_button, self.model, self.device, self.microphone, self.refresh,
                       self.thinking,self.api_settings,self.punctuation_mode,self.glossary_action,
                       self.folder, self.browse, self.threshold, self.precision, self.correction_model, self.llm_precision, self.comparison_precision, self.writable_window, self.load_button):
            widget.setEnabled(not busy)
        self.stop.setEnabled(busy)
        if self.llm_loader and self.llm_loader.isRunning():
            self.correction_model.setEnabled(False)
            self.llm_precision.setEnabled(False)
            self.comparison_precision.setEnabled(False)
        if self.correction_model.currentData()=='deepseek':
            self.thinking.setEnabled(False)
            self.llm_precision.setEnabled(False)

    def import_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入录音文件', '',
                    '音频文件 (*.wav *.flac *.ogg *.mp3 *.aiff *.aif);;所有文件 (*)')
        if path:
            self.begin(path)

    def begin(self, input_path=None):
        if ((self.session and self.session.isRunning()) or
                (self.loader and self.loader.isRunning()) or
                (self.llm_loader and self.llm_loader.isRunning())):
            return
        if (self.host.engine is None or self.host.options != (self.device.currentData(), self.precision.currentData(), self.config) or
                (self.enable_correction.isChecked() and (not self.correction_selection_matches() or not self.local_backend.ready))):
            self.state.setText('所选模型尚未就绪 · 请先点击 Load model；仅用 ASR 可关闭本地纠错。')
            return
        if (not input_path and self.microphone.currentData() is None) or not self.folder.text().strip():
            QMessageBox.warning(self, '无法开始', '请选择麦克风和保存目录。')
            return
        if self.session:
            self.session.retire()
        self.output.clear()
        self.comparison_output.clear()
        self.set_busy(True)
        self.session = Session(self.device.currentData(), self.precision.currentData(),
                               self.microphone.currentData(),
                               self.folder.text(), self.threshold.value(), self.host, self.config, self.local_backend, input_path,
                               window_chars=self.writable_window.value(),punctuation_mode=self.punctuation_mode.currentData())
        self.session.status.connect(self.state.setText)
        self.session.text.connect(self.refresh_transcript)
        self.session.level.connect(lambda rms: self.meter.setValue(min(100, int(rms * 500))))
        self.session.saved.connect(self.show_saved_paths)
        self.session.error.connect(self.failed)
        self.session.finished.connect(self.finished)
        self.session.start()

    def show_saved_paths(self, paths):
        self.path_label.setToolTip(paths)
        self.path_label.setText('对比结果：分别保存 _2b 与 _0_8b 数据库和 TXT（悬停查看路径）'
                               if ' | ' in paths else '转录数据库：' + Path(paths).name)

    def end(self):
        if self.session and self.session.isRunning():
            self.session.request_stop()
            self.stop.setEnabled(False)
            self.state.setText('正在停止输入并处理剩余语音，请稍候…')

    def failed(self, message):
        self.state.setText('发生错误 · 已停止')
        QMessageBox.critical(self, '语音转写失败', message + '\n详细日志：logs/pipeline.log')

    def finished(self):
        self.set_busy(False)
        self.meter.setValue(0)
        self.refresh_transcript()
        if self.closing:
            self.close()

    def closeEvent(self, event):
        if self.llm_loader and self.llm_loader.isRunning():
            self.closing = True
            self.local_backend.cancel.set()
            if self.session and self.session.isRunning():
                self.session.request_stop('shutdown_flush')
            self.state.setText('正在等待本地模型加载结束并保存会话…')
            event.ignore()
            return
        if self.loader and self.loader.isRunning():
            self.closing = True
            self.state.setText('等待模型加载完成后退出…')
            event.ignore()
            return
        if self.session and self.session.isRunning():
            self.closing = True
            self.local_backend.cancel.set()
            self.session.request_stop('shutdown_flush')
            self.stop.setEnabled(False)
            self.state.setText('正在处理剩余音频并保存，完成后自动退出…')
            event.ignore()
        else:
            if self.session:
                self.session.retire()
            self.host.close()
            self.local_backend.close()
            event.accept()


if __name__ == '__main__':
    configure_logging(ROOT / 'logs')
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
