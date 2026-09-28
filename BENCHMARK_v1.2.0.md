# v1.2.0 本地 Qwen 集成基线

**结论：运行链路已接入；纠错质量尚未通过完整验收。** BF16 漏掉 A/D，INT4 漏掉 D。小样本不能代表整体准确率。

机器：RTX 4060 Laptop 8GB；Python 3.10.11、Torch 2.5.1+cu121、FunASR 1.2.7、Transformers 5.17.0。
提示版本：`qwen_asr_correction_v5`；默认 BF16、唯一锚点；ASR GPU FP16、beam=3/nbest=3。

## 显存与功率

| 阶段 | PyTorch allocated MiB | PyTorch reserved MiB | 整卡峰值 MiB | 平均 GPU W | 峰值 GPU W |
|---|---:|---:|---:|---:|---:|
| 无模型/CUDA已初始化 | 0.0 | 0.0 | 0 | 8.91 | 12.85 |
| 仅ASR常驻 | 850.6 | 862.0 | 946 | 12.47 | 12.64 |
| 仅Qwen BF16常驻 | 3597.4 | 3602.0 | 3718 | 9.78 | 12.35 |
| ASR+Qwen常驻 | 4448.0 | 4464.0 | 4586 | 20.45 | 22.77 |
| ASR推理 | 1142.4 | 1162.0 | 1280 | 14.05 | 17.22 |
| Qwen推理 | 3683.0 | 3714.0 | 3836 | 39.76 | 45.59 |
| ASR/Qwen并发 | 4845.6 | 4818.0 | 5022 | 39.27 | 43.52 |

推理阶段各持续约10秒，独立于10分钟稳定性回放，减轻GPU功率传感器滞后影响。推理行的 allocated 为阶段内分配峰值，reserved为阶段末值；常驻功率仅取加载后2秒，不代表长期待机。整卡数据含CUDA上下文、驱动和其他程序，并非模型权重大小。

## BF16 / NF4 对照（相同8类案例，各重复2次）

| 指标 | BF16 | INT4/NF4 |
|---|---:|---:|
| 常驻 allocated MiB | 3589.3 | 1646.8 |
| 推理整卡峰值 MiB | 3954 | 2000 |
| 请求时延 median / p95 秒 | 0.413 / 0.600 | 0.460 / 1.215 |
| 近似 prefill/TTFT median 秒 | 0.132 | 0.140 |
| decode median 秒 | 0.278 | 0.320 |
| 输入 / 输出 tokens 范围 | [439, 845] / [12, 12] | [439, 845] / [12, 33] |
| 请求内平均 / 峰值 GPU W | 38.50 / 42.21 | 40.17 / 42.66 |
| 每完成请求近似 J（含no-op） | 16.54 | 21.35 |
| 符合案例期望 | 12 / 16 | 14 / 16 |

200ms固定频率采样nvidia-smi功率，梯形积分并以端点矩形补齐边界；是整卡近似能量，不是精确模型能耗或整机能耗。短请求样本少；不同输出长度影响能耗，低显存不等于低能耗。
TTFT由streamer首token时间近似，包含第一token计算及同步开销；不是独立CUDA prefill kernel计时。每种精度包含第一次请求，p95采用nearest-rank，小样本仅作基线。
BF16没有成功修改A/D，不能把no-op能耗当成成功纠错能耗。INT4在A成功，并不说明量化普遍改善质量。

## 模型专项结果

| 案例 | BF16 | INT4 |
|---|---|---|
| A 上下文/N-best型号纠正 | 漏改 | 通过 |
| B 原文正确 | 通过 | 通过 |
| C 缺乏消歧证据 | 通过 | 通过 |
| D 后文回改 | 漏改 | 漏改 |
| E 禁止润色 | 通过 | 通过 |
| F 保留口语 | 通过 | 通过 |
| G 冻结内容 | 保持不变 | 保持不变 |
| H 会议内提示注入 | 保持不变 | 保持不变 |

G另有强制越界/冻结锚点测试，程序拒绝；H单例不能证明普遍注入免疫。原有窗口/修订/长度限制保持不变，它们不证明语义正确。
相对偏移协议在v5只有5/8通过，A/C返回无效结构；锚点免除模型数位置负担。v4曾错误反向改写型号和无依据替换，因此未采用该激进提示。

## 10分钟稳定性

- 实际运行600.1秒，60段真实Paraformer推理，LLM结果：{'INVALID_OUTPUT': 3, 'NO_CHANGE': 57}。
- 最大待执行队列0；stale比例0.0%；Qwen加载1次，回放期间ASR未重载。
- SQLite integrity_check=ok，UI与Store一致；UI事件循环p95间隔0.032秒，最大0.110秒。
- 采样 allocated 4464.3–4748.5 MiB；RSS 1696.3–1935.1 MiB。
- 每10秒回放一次官方样例，使用单调样本时间轴；Qt为offscreen实际布局/事件循环，不是物理显示器帧率。
- 这是基础稳定性测试，不是多人会议准确率、最大吞吐压力或多小时内存泄漏证明。

## 依赖、复现与限制

新增核心包：transformers5.17.0、accelerate1.15.0、safetensors0.8.0、tokenizers0.23.2、huggingface-hub1.33.0、psutil7.2.2；可选bitsandbytes0.50.2。传递依赖见requirements-lock.txt，已有Torch/FunASR版本未变，pip check通过。
未安装causal_conv1d/flash-linear-attention，采用Transformers正确的PyTorch参考实现。未引入新虚拟环境、在线LLM、视觉、ONNX、NPU或流式ASR。
下载使用官方ModelScope国内源，约4分25秒完成4.55GB；safetensors SHA256与固定HF revision一致。应用本地加载官方文本类，实际文本参数1,881,825,088，忽略视觉塔。
启动配置和完整复现命令见README。原始测量在logs/，不提交会话音频、转录或完整提示；可分享汇总为benchmarks/v1.2.0_summary.json。
63项自动测试通过，包括真实后端OOM/timeout/runtime异常模拟、严格解析、锚点、陈旧响应、冻结边界和配置错误降级；五种ASR精度/设备路径在增量依赖安装后通过真实推理。
已知限制：纠错召回不足，语义正确性无法由PatchValidator保证；超时只在生成步之间检查，不能中断挂住的CUDA内核；1024字符窗口外不回改。

## 真实麦克风测试

- Realtek内置麦克风，30.2秒，阈值0.001，ASR FP16 + Qwen INT4。
- 2个非空ASR片段，2次Qwen请求；结果{'INVALID_OUTPUT': 1, 'NO_CHANGE': 1}，无音频丢弃或设备溢出。
- 本次没有接受的LLM修改：一次no-op，一次无效输出被严格拒绝。证明真实采集/Evidence/LLM/Store链路运行，不能证明纠错质量达标。
- 初次0.008阈值只触发空识别；降低至0.001后有效触发。实际中文聊天包含少量英文字母候选，没有人工逐字真值，未计算WER/CER，也没有完整覆盖全部建议技术词。
- 原始RF64 WAV、N-best、请求结果和TXT保存在本地recordings/、transcripts/smoke/；报告不包含真实聊天文字。

## 文件变更

- 本地模型与协议：qwen_backend.py、qwen_spec.py、download_qwen.py、llm_protocol.py、prompts/asr_correction_system.txt、settings.py。
- 集成与审计：app.py、llm_pipeline.py、observability.py；patches.py和transcript_store.py只增加请求元数据透传，不改变核心校验算法。
- 验证与工具：qwen_cases.py、test_qwen.py、test_ui.py、verify_qwen.py、benchmark_local.py、gpu_telemetry.py、smoke_microphone.py、inspect_session.py、summarize_benchmark.py。
- 环境与文档：requirements-llm.txt、requirements-llm-int4.txt、requirements-lock.txt、version.py、README.md、ARCHITECTURE.md、CHANGELOG.md及本报告/脱敏测量汇总。
