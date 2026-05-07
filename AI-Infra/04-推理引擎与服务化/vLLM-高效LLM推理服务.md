# vLLM：高效 LLM 推理服务

> vLLM 是 UC Berkeley Sky Computing Lab 发起的开源 LLM 推理引擎，以 PagedAttention 技术为核心，实现了业界领先的推理吞吐量，支持 200+ 模型架构。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心技术](#2-核心技术)
- [3. PagedAttention 详解](#3-pagedattention-详解)
- [4. 分布式推理](#4-分布式推理)
- [5. 量化与优化](#5-量化与优化)
- [6. 生态与模型支持](#6-生态与模型支持)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **发起方** | UC Berkeley Sky Computing Lab |
| **核心定位** | 快速、易用、低成本的 LLM 推理服务 |
| **开源协议** | Apache 2.0 |
| **社区规模** | 2000+ 贡献者，最活跃的推理项目之一 |
| **官网** | [vllm.ai](https://vllm.ai) |

---

## 2. 核心技术

### 2.1 内存管理

- **PagedAttention**：将 KV Cache 分页管理，显存利用率从 ~60% 提升至接近 100%
- **Prefix Caching**：前缀共享缓存，重复前缀请求复用 KV Cache
- **Chunked Prefill**：将长 prefill 序列分块，与 decode 交错执行

### 2.2 批处理与调度

- **Continuous Batching**：请求动态加入/离开 batch，无需等待整批完成
- **Disaggregated Prefill/Decode**：PD 分离部署，prefill 和 decode 分别优化
- **Speculative Decoding**：n-gram、EAGLE、DFlash 等投机解码策略

### 2.3 执行优化

- **CUDA/HIP Graphs**：减少 CPU 启动开销
- **torch.compile**：自动 Kernel 生成和图级优化
- **Attention Backends**：FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA

---

## 3. PagedAttention 详解

### 3.1 问题背景

传统推理为每个请求预分配连续的 KV Cache 内存块，类似于操作系统中的连续内存分配问题：
- 内存碎片严重
- 实际利用率仅约 60%
- 无法动态扩展序列长度

### 3.2 PagedAttention 原理

借鉴操作系统虚拟内存的分页思想：

```
传统方式：每个请求 → 预分配连续大块显存
PagedAttention：KV Cache → 固定大小的 Block（Page）→ 按需分配 → 非连续存储
```

- **Block Size**：通常为 16、32 或 64 tokens
- **Block Table**：每个序列维护一个 Block 映射表
- **Copy-on-Write**：共享前缀的序列共享 Block，写入时复制

### 3.3 效果

- 显存利用率：**~60% → 接近 100%**
- 支持的并发请求数：提升 2-4 倍
- 吞吐量提升：2-4 倍（相比未优化引擎）

---

## 4. 分布式推理

vLLM 支持多种并行策略，适配从单卡到多机集群：

| 并行类型 | 说明 |
|----------|------|
| **张量并行（TP）** | 单层切分到多张 GPU |
| **流水线并行（PP）** | 模型按层分段 |
| **数据并行（DP）** | 多副本服务不同请求 |
| **专家并行（EP）** | MoE 模型专家分发 |
| **上下文并行（CP）** | 长序列切分 |

---

## 5. 量化与优化

vLLM 支持业界最广泛的量化方案：

| 量化类型 | 说明 |
|----------|------|
| **FP8** | Hopper 架构原生支持 |
| **INT8 / INT4** | 通用整数量化 |
| **GPTQ / AWQ** | 权重-only 量化 |
| **GGUF** | llama.cpp 格式兼容 |
| **MXFP8 / MXFP4 / NVFP4** | 微缩浮点格式 |
| **ModelOpt / TorchAO** | NVIDIA 和 PyTorch 量化工具 |

---

## 6. 生态与模型支持

### 6.1 支持的模型类型

- **Decoder-only LLM**：Llama、Qwen、Gemma、DeepSeek 等
- **MoE 模型**：Mixtral、DeepSeek-V3、Qwen-MoE
- **多模态模型**：LLaVA、Qwen-VL、Pixtral
- **Embedding 模型**：E5-Mistral、GTE、ColBERT
- **Reward/分类模型**：Qwen-Math 等

### 6.2 硬件支持

- NVIDIA GPUs（主要优化目标）
- AMD GPUs（ROCm）
- Intel CPUs、Google TPUs
- Huawei Ascend、Apple Silicon 等（社区支持）

### 6.3 API 兼容性

- **OpenAI-compatible API**：直接替换 OpenAI 服务
- **Anthropic Messages API**
- **gRPC 支持**

---

## 7. 推理引擎全景对比

| 引擎 | 特点 | 适用场景 |
|------|------|----------|
| **TensorRT** | NVIDIA 专用、极致优化（算子融合/精度校准/Kernel 自动调优） | 生产推理 |
| **ONNX Runtime** | 跨平台、通用，多框架支持 | 多框架互操作 |
| **vLLM** | PagedAttention、高吞吐 | LLM 服务 |
| **SGLang** | RadixAttention、PD 解耦、高吞吐 | 前沿推理/大规模部署 |
| **Triton Inference Server** | 服务化、多模型混部 | 企业部署 |
| **llama.cpp** | 纯 C/C++、零依赖、极致量化 | 边缘推理/本地部署 |

## 8. 推理引擎架构基础

现代推理引擎的通用优化手段：

- **模型图优化**：计算图简化、死代码消除
- **算子融合**：减少 Kernel 启动开销（如 Attention+Residual+LayerNorm 融合）
- **常量折叠**：预计算静态张量，消除运行时开销
- **内存规划**：显存生命周期分析、内存复用、碎片整理

---

> 💡 **核心要点**：vLLM 是**开源 LLM 推理的标杆项目**。其 PagedAttention 技术解决了 KV Cache 显存管理的核心痛点，Continuous Batching 和丰富的量化支持使其成为生产部署的首选。对于 AI Infra 工程师，深入理解 vLLM 的调度器、内存管理和分布式推理配置是必备技能。
