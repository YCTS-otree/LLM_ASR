# LLM_ASR

本地会议转录工具：**FunASR Paraformer-large + Qwen3.5 上下文纠错与标点恢复**。支持麦克风持续录音、录音文件离线转录、2B / 0.8B 双模型对比，以及可追溯的原始识别和修改历史。

**版本：1.5.1 · 源代码许可：[GPL-3.0-only](LICENSE)**

这是实验性桌面应用。优先忠实保留讲话内容；模型仍可能漏标点、误断句或误改词，需要人工校对。当前使用离线 Paraformer 按声音停顿分段，不是 FunASR streaming 模型，没有说话人分离，也不保证实时响应。

**质量状态：尚未达到会议转录验收标准。** v1.5.0增加上下文回改、独立标点和在线API，但允许改词不等于改词正确。真实样本与限制见 [质量报告](BENCHMARK_v1.5.0.md)；历史漏首段问题见 [v1.4.1报告](BENCHMARK_v1.4.1.md)。

## 功能

- 麦克风录音与本地 WAV / FLAC / OGG / MP3 / AIFF 文件导入，格式以 SoundFile/libsndfile 能解码为准。
- 参数调整完成后点击 **Load model** 统一加载。启动、修改参数、重新勾选纠错均不自动加载；相同配置可复用常驻模型。
- 单独选择 Qwen3.5-2B 或 Qwen3.5-0.8B，支持 BF16、FP16、可选 INT4/NF4。
- 双模型并发：一次 ASR、两份独立上下文和转录结果，两栏显示并分别保存。
- 可修改窗口 **256–4096 个 Unicode 字符**，默认 **2048**。新内容到来后重新校对整个窗口，包括已结束的旧句；已冻结历史不解冻。
- 每个写入目标最多128字，优先沿已有标点或英文单词边界划分，读取整个窗口的前后文。读上下文重叠、写入范围不重叠，不重复拼接文字。
- 允许上下文支持的同音纠错、数字/年份/术语规范化，不要求正确词出现在 N-best 中。基础 CT-Transformer 标点独立运行，关闭或拒绝 LLM 时仍保留基础标点。
- Qwen 思考模式可选；普通生成最多1024 tokens，思考模式4096 tokens，整批最多600秒。推理过程不进入转录或日志。

Qwen思考模式默认关闭：本机测试出现4096 tokens内仍未结束思考的情况，增加等待时间不保证更准。官方模型卡也说明2B可能发生思考循环，见[Qwen3.5-2B模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)。未完成的输出不会写入转录。

- 可选 **DeepSeek 在线 API**，支持模型 ID 和思考开关；无需安装 OpenAI SDK。
- SQLite 保存不可变 ASR、N-best、修订、拒绝原因及原始音频位置；停止或文件结束时导出 UTF-8 TXT。
- 模型从国内 ModelScope 显式下载；选择本地 Qwen 时不调用在线 LLM。

修改窗口内的旧句可结合后文回改，例如后文明确英文拼写后再修正前文。原始 ASR 和每次修订保留。扩大窗口不会自动修复已经结束的旧会话，也不保证纠错更准。音频仍采用最长15秒分段；文本重叠上下文不能恢复已经漏识别的声音。

## 系统与硬件要求

| 项目 | 要求 / 验证情况 |
|---|---|
| 操作系统 | 已验证 Windows 11 x64；其他平台未做完整验收 |
| Python | **CPython 3.10 x64**，实测 3.10.11；建议使用相同版本复现依赖 |
| 仅 ASR | CPU 可用 FP32 或 INT8；NVIDIA CUDA GPU 可用 FP16 / BF16 / FP32 |
| 本地 Qwen | 当前后端要求 **NVIDIA CUDA GPU**，未实现 CPU / AMD GPU / NPU 后端 |
| 显存 | 已验证 **RTX 4060 Laptop 8GB**；建议 8GB 或更多，非最低显存承诺 |
| 精度 | 默认 2B BF16；双模型默认 2B NF4 + 0.8B BF16。BF16 需硬件支持；FP16 可手动选择 |
| 内存 | 建议 16GB 以上，双模型推荐 32GB；这是容量建议，未验证最低配置 |
| 硬盘 | 建议至少 20GB 可用空间，另留录音空间；模型约 0.88GB + 4.55GB（2B）+ 1.75GB（0.8B） |
| 音频输入 | 麦克风模式需要可用输入设备；文件导入不需要麦克风 |

双模型实测自身 CUDA 分配峰值约 **4241 MiB**，并不等于系统总显存需求。其他程序、CUDA 缓存、输入长度和精度都会影响占用；显存不足时可关闭其他 GPU 工作、选择 INT4 或关闭本地纠错。该值来自 v1.3.0 短样本，见 [测量报告](BENCHMARK_v1.3.0.md)。

依赖使用 PyTorch CUDA 12.1 构建，需要兼容的 NVIDIA 驱动。常规 wheel 安装不要求单独安装 CUDA Toolkit、C++ 编译器或 NPU SDK。本项目没有安装可选的 `causal_conv1d` / `flash-linear-attention` 加速内核，采用参考 PyTorch 实现，速度较慢。

RF64 音频按原始采样率保存为单声道 PCM16，例如 44.1kHz 约 **318MB/小时**（计算值）。导入文件也会生成音频副本；长录音需要额外磁盘空间。

## 库要求

| 库 | 版本 / 用途 |
|---|---|
| PySide6 | `>=6.11,<7`，测试 6.11.2，Qt 界面 |
| FunASR | `1.2.7`，Paraformer 推理与 N-best |
| torch / torchaudio | `2.5.1+cu121`，ASR 与 LLM 共用 |
| NumPy / SciPy | `numpy<2`、`scipy>=1.10,<2`，音频数组与重采样 |
| sounddevice / soundfile | `>=0.5,<0.6` / `>=0.13,<0.14`，录音及文件解码 |
| ModelScope | `>=1.20,<2`，国内模型下载 |
| Transformers | `5.17.0`，Qwen3.5 文本后端 |
| accelerate / safetensors | `1.15.0` / `0.8.0` |
| tokenizers / huggingface-hub | `0.23.2` / `1.33.0` |
| bitsandbytes | **可选** `0.50.2`，仅 INT4/NF4 需要 |
| psutil | `7.2.2`，测量工具 |

基础依赖见 [requirements.txt](requirements.txt)，LLM 增量见 [requirements-llm.txt](requirements-llm.txt)，量化见 [requirements-llm-int4.txt](requirements-llm-int4.txt)。[requirements-lock.txt](requirements-lock.txt) 是测试环境的固定版本约束，不是跨平台兼容保证。

## 安装与启动

先安装 Python 3.10 x64 与 NVIDIA 驱动。下列为 **PowerShell** 示例；已有合适 Python 环境时可直接复用，不必再建环境。

```powershell
git clone https://github.com/YCTS-otree/LLM_ASR.git
cd LLM_ASR

# 新用户可选择项目内环境；不要求更改全局 Python
py -3.10 -m venv .venv
$pythonExe = ".\.venv\Scripts\python.exe"

# 包使用清华源；CUDA wheel 仍由 requirements 中的 PyTorch 官方源提供
& $pythonExe -m pip install -r requirements-llm.txt -c requirements-lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 若需要 INT4 或默认双模型组合，再安装可选量化库
& $pythonExe -m pip install -r requirements-llm-int4.txt -c requirements-lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 显式下载官方模型，默认 ModelScope 国内源
& $pythonExe download_models.py
& $pythonExe download_qwen.py --download --model 2b
& $pythonExe download_qwen.py --download --model 0.8b

& $pythonExe app.py
```

只使用 2B 时不用下载 0.8B，反之亦然。仅 ASR 或 DeepSeek 可只安装 `requirements.txt`、下载 Paraformer 与基础标点，并关闭本地 Qwen；CPU 需选择 FP32 或 INT8。基础标点约292MiB，使用CPU。已有 Paraformer 的用户可执行 `python download_models.py --punctuation-only`，只下载新增标点权重。

`start.bat` 优先使用项目 `.venv`，其次使用当前用户的标准 Python310 安装路径，最后使用 PATH 中的 `python`。其他自定义环境请用该环境的 `python app.py` 启动。

Qwen 下载脚本校验固定 SHA256，模型标识和对应官方 revision 在 [qwen_spec.py](qwen_spec.py)。国内仓库文件若更新导致校验失败，请勿跳过校验，可改用 `--source huggingface` 下载固定 revision。程序启动不会自动下载权重。

## 使用方法

1. 先选择输入设备、模型、精度与修改窗口，再点击 **Load model**。等待加载完成后开始；改变模型或精度后需要再次点击按钮应用。仅修改阈值、目录或窗口大小不需重载权重。
2. 点击「开始录音」，或点击「导入录音」选择已有文件。
3. 如讲话较轻可调低声音触发阈值；停顿约 0.8 秒结束一段，连续讲话最多 15 秒切分。硬切不等于句子结束。
4. 点击「停止并保存」后，等待已排队的语音与纠错处理结束；导入到文件末尾会自动保存。
5. 在保存目录查看 TXT，SQLite 中保留原始 ASR 与所有修改依据。界面「若干处未通过」表示有短目标未通过检查，需要人工查看原文。

麦克风和导入互斥。导入采用背压逐段等待，适合不追求实时的完整处理；读取百分比达到 100% 时仍可能有推理未结束。麦克风模式仍为异步、有界队列，计算落后时可能合并纠错请求或丢弃待 ASR 片段，均记录审计。对完整性要求高时，保留录音并使用文件导入重转。

双模型模式左侧 2B、右侧 0.8B，使用同一份 ASR 证据，但后续上下文各自演化。它比较的是两条转录流程，不能把所有后续请求当作完全相同的模型输入。并发共享 GPU，不保证更快。0.8B 尤其 NF4 在既有测试中质量较弱，见 [对比报告](BENCHMARK_v1.3.0.md)。

插入 USB 麦克风后，停止录音再点击「刷新」。同一个麦克风可能显示多个 PortAudio 接口。关闭纠错不影响继续采集和 ASR；同配置再次开始会复用模型。

## 数据保存与隐私

使用 DeepSeek：选择「DeepSeek 在线 API」→「DeepSeek 设置」→填写服务商 API Key 和模型 ID→点击 Load model→导入或录音。当前官方文档列出的模型 ID 为 `deepseek-flash`、`deepseek-v4-pro`，界面允许自定义，以[官方接口文档](https://api-docs.deepseek.com/api/create-chat-completion/)为准。思考默认开启。Load model仅检查本地配置，真实鉴权在首次请求进行；HTTP失败显示在会话事件中，不自动重试计费请求。

在线模式发送转录文本、识别候选和上下文，**不上传音频文件**。密钥输入框留空时，点击 **Load model** 会读取当前运行目录（进程工作目录）下的 `DEEPSEEK.key`，内容为单个UTF-8密钥文本，允许BOM和首尾空白。文件不存在时提示错误，不会自动创建。修改文件后再次点击按钮会重新读取。使用 `start.bat` 时工作目录就是该脚本所在目录；从终端启动则使用终端当前目录。手动输入的密钥优先于文件。

程序只在内存中使用密钥，不回写文件、不写入日志或SQLite，也不回填文件密钥到界面。用户自行创建的密钥文件由用户管理，`*.key` 已被Git忽略。请求固定使用 `https://api.deepseek.com/chat/completions`，拒绝重定向。API计费与数据处理遵循服务商条款。未提供密钥时仅验证了模拟响应，未完成真实在线服务验收。

| 目录 | 内容 |
|---|---|
| `models/` | 下载的模型权重、配置、缓存 |
| `recordings/` | RF64 WAV 原始单声道音频副本 |
| `transcripts/` | SQLite、TXT、离线回放结果 |
| `logs/` | 轮转日志和本地验证输出 |

这些目录均由 `.gitignore` 排除。错误日志可能包含本地路径，本地Qwen的 debug 日志会记录提示词和最终答案；请勿公开真实录音、会话数据库或日志。DeepSeek不记录请求正文、原始响应或推理内容。模型下载与依赖安装需要联网；本地模式推理使用本地文件。

## 验证与开发

```powershell
& $pythonExe -m unittest discover -q
& $pythonExe verify_ui.py

# 在新数据库回放既有 ASR 证据，不覆盖源会话、不重新运行 ASR
& $pythonExe replay_evidence.py transcripts\your_session.sqlite3 --window-chars 2048
```

短目标输出会转为精确 Patch，再检查修订号、冻结区、操作数量/跨度和重叠。大段复制、非重复内容的纯删除会被拒绝；允许有界的上下文替换。失败保留当前文本和基础标点。规则不能证明改词语义正确，仍需人工复核。

参见 [架构](ARCHITECTURE.md)、[本版回归记录](BENCHMARK_v1.5.0.md)、[更新记录](CHANGELOG.md)。旧报告描述各自版本的实验条件，不能视为当前版在任意音频上的准确率。

## 许可与第三方组件

本项目原创源代码采用 **GNU GPL v3.0 only**，完整许可见 [LICENSE](LICENSE)。第三方库和模型遵循各自许可证，不因本仓库 GPL 标记而改变。仓库不附带模型权重、Python 环境或用户录音。

- [FunASR](https://github.com/modelscope/FunASR) 与 [Paraformer 模型页](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)。请分别查看代码许可和模型使用条件。
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) / [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)：官方模型卡标示 Apache-2.0。
- [CT-Transformer基础标点模型](https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/)：ModelScope模型页标示Apache-2.0，权重不随源码分发。
- [PySide6 / Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html) 及其他安装依赖的许可由各项目提供；再分发打包程序时需同时保留相应许可和通知。
