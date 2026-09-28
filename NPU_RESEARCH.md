# FunASR 在本机 AMD NPU 上运行的可行性

调研日期：2026-09-26。仅查询资料和设备信息，未安装 NPU SDK、未导出/编译 NPU 模型、未执行 NPU 推理。

## 本机硬件

- CPU：AMD Ryzen AI 9 HX 370（Strix / Ryzen AI 300）。
- Windows 设备：`NPU Compute Accelerator Device`，`PCI\VEN_1022&DEV_17F0`，状态 OK。
- NVIDIA 独显：RTX 4060 Laptop GPU，8GB；它与 AMD NPU 是两套独立硬件。

AMD 官方产品说明标注该处理器 NPU 算力最高 50 TOPS。TOPS 不能直接换算为 Paraformer 转写速度。[AMD 产品页](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html)

## 结论

**当前应用不能直接切换到 AMD NPU。存在工程适配路线，但不能宣称 Paraformer-large 已经兼容或一定比 RTX 4060 更快。**

现有 FunASR AutoModel 使用 PyTorch 的 CPU/CUDA 路径。AMD 官方 Ryzen AI 使用 ONNX Runtime 和 Vitis AI Execution Provider 部署 NPU 模型；不是给 PyTorch 的 `device` 填入 `npu` 即可使用。[AMD 开发流程](https://ryzenai.docs.amd.com/en/latest/)

截至 Ryzen AI Software 1.8，Strix 平台支持 BF16 CNN/NLP 模型路径。部署时 Vitis AI EP 把支持的子图放在 NPU，其余部分留在 CPU。因此“成功加载模型”和“全模型在 NPU 上运行”必须分别验证。[兼容平台](https://ryzenai.docs.amd.com/en/latest/relnotes.html)、[编译与部署](https://ryzenai.docs.amd.com/en/latest/modelrun.html)

FunASR 上游提供 Paraformer ONNX 导出及 ONNX Runtime 推理，但我查到的文档没有给出本机 AMD NPU 专用的 Paraformer 220M 即用模型或验证结果；AMD 1.8 发布说明和示例目录也未列出 Paraformer。[FunASR ONNX Runtime](https://github.com/modelscope/FunASR/tree/main/runtime/python/onnxruntime)、[AMD 示例](https://ryzenai.docs.amd.com/en/latest/examples.html)

## 若未来需要实施

以下是基于上述工具链的适配推断，尚未验证：

1. 导出 Paraformer 的 ONNX 图，核对音频长度、预测输出长度及 CIF 等运算的导出行为。
2. 对照 Ryzen AI 的算子支持与形状要求，确定是否需要固定长度、按长度分桶或拆分编码器/解码器。
3. 选择 BF16 编译路径或适合 NPU 的校准量化格式。当前应用的 PyTorch CPU 动态 INT8 不能直接当作 NPU 模型。
4. 使用 Vitis AI EP 编译，检查每个子图实际运行设备和 CPU 回退比例。
5. 对比中文字符错误率、端到端延迟、功耗和峰值内存，再判断相对 CUDA 的价值。

适配能否成功、NPU 覆盖率和性能，都需要真实编译与测试才能确定。本次没有实施这些步骤。
