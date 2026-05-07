# Triton 高性能推理框架开发（课程体系）

> 从零构建高性能 LLM 推理框架的 Triton 版本，25 课时覆盖 Triton 基础、Attention 实现、显存管理、多模型支持。

---

## 1. 从零构建高性能推理框架 — Triton 版本（25 课时）

| 阶段 | 课时 | 内容 |
|------|------|------|
| **Triton 基础** | 课时1 | Triton 内核开发基础上（Triton/CUDA/PyTorch 语法对比、保留关键字、张量概念） |
| | 课时2 | Triton 内核开发基础下（tl.load/tl.store/tl.program_id、调试方法） |
| | 课时3 | Triton 编译基础（ttir→tgir→LLVM IR→ptx→cubin 编译流程、与 CUDA 编译对比） |
| **核心算子** | 课时4 | 动手写 Triton Softmax 算子（线性算子写法、内核调用流程、分块并行加速） |
| | 课时5 | 动手写 Triton Matmul 算子（二维分块、tl.dot 基础接口） |
| | 课时6 | 优化 Matmul 算子（分组技术优化、效率逼近甚至超越 CuBlas） |
| | 课时7 | 实现 DeepSeek 中的 fp8 gemm 算子（分组量化在 Triton 中的实现） |
| | 课时8 | 用 Triton 实现大模型中的 MLP 算子（串联激活函数、Matmul 等算子） |
| **Attention** | 课时9 | FlashAttention 的实现（单头/多头注意力、Online-Softmax、Triton 实现加速） |
| | 课时10 | FlashAttentionV2 实现（V1 和 V2 公式和实现区别、主要用于 prefill 阶段） |
| | 课时11 | FlashAttentionV3 实现（V3 和 V1/V2 区别、主要用于 decode 阶段） |
| **显存管理** | 课时12 | 大模型显存分块管理机制（PagedAttention）（PageAttention 基础、按需节省显存） |
| | 课时13 | 分组注意力机制实现（GQA 节省显存使用、Triton 实现形式） |
| | 课时16 | 分块显存的管理和动态更新（动态推理中记录管理 kv cache 显存块用量、req_table） |
| **高级特性** | 课时14 | 分析 Triton 知名开源项目 Flaggems（归约算子分类、写法模板化） |
| | 课时15 | Triton 融合算子（rmsnorm）（一个 Triton 内核中组合多个不同计算过程） |
| | 课时17 | RoPE 的实现（位置相关的旋转操作、捕捉序列顺序关系） |
| | 课时18 | 温度系数 top-p 和采样策略（调整关键参数控制生成文本多样性和创造性） |
| **模型支持** | 课时19 | hf 权重的加载和解析（直接加载转换 HuggingFace 大模型如 Qwen3） |
| | 课时20 | 模型执行器模块分析 |
| | 课时21 | Llava 多模态推理流程（扩展至多模态推理、图像-文本联合输入） |
| | 课时22 | Llama3.2 和 Qwen3 的区别（注意力模块、位置编码、kv cache 更新实现区别） |
| | 课时23 | LLM 参数量和计算量分析（理论显存占用量和理论运行时间） |
| **项目总结** | 课时24 | DeepSeek MOE 的结构和实现 |
| | 课时25 | 代码走读（从全局角度分析《自制大模型推理框架》课程项目） |

---

> 🔑 **核心要点**：Triton 版本的推理框架课程聚焦于"用 Python 写高效 GPU Kernel"。从 Softmax 到 FlashAttention，从 PagedAttention 到 MoE，完整覆盖一个生产级推理框架的所有核心算子。
