# SGLang：高性能推理框架

> SGLang 是 LMSYS 开源的高性能大模型服务框架，以 RadixAttention、零开销 CPU 调度器、PD 解耦为核心特性，已在全球 40 万+ GPU 上部署，日处理万亿级 tokens。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心技术](#2-核心技术)
- [3. RadixAttention 详解](#3-radixattention-详解)
- [4. 大规模部署能力](#4-大规模部署能力)
- [5. RL 与后训练支持](#5-rl-与后训练支持)
- [6. 硬件与模型生态](#6-硬件与模型生态)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **开发者** | LMSYS ( Large Model Systems Organization ) |
| **核心定位** | 高性能 LLM 与多模态模型服务框架 |
| **部署规模** | 40 万+ GPU，日处理万亿级 tokens |
| **开源协议** | Apache 2.0 |
| **文档** | [docs.sglang.io](https://docs.sglang.io/) |

### 行业采用

xAI、AMD、NVIDIA、Intel、LinkedIn、Cursor、Oracle Cloud、Google Cloud、Microsoft Azure、AWS、MIT、Stanford、UC Berkeley、清华大学等。

---

## 2. 核心技术

### 2.1 Fast Runtime

| 特性 | 说明 |
|------|------|
| **RadixAttention** | 前缀缓存，Radix Tree 管理 KV Cache |
| **Zero-Overhead CPU Scheduler** | CPU 调度零开销，全异步流水线 |
| **Prefill-Decode Disaggregation** | PD 分离，分别优化吞吐和延迟 |
| **Speculative Decoding** | 投机解码加速 |
| **Continuous Batching** | 连续批处理 |
| **Chunked Prefill** | 分块 prefill |
| **Structured Outputs** | 基于 FSM 的结构化输出（JSON 等） |
| **Multi-LoRA Batching** | 多 LoRA 适配器批处理 |

### 2.2 并行策略

- Tensor Parallelism
- Pipeline Parallelism
- Expert Parallelism（大规模 MoE）
- Data Parallelism

---

## 3. RadixAttention 详解

### 3.1 问题背景

多轮对话、批量推理等场景中，大量请求共享相同前缀（如 system prompt、 Few-shot examples）。传统 PagedAttention 虽支持 Block 级共享，但缺乏高效的前缀匹配机制。

### 3.2 RadixAttention 原理

基于 **Radix Tree（基数树）** 管理 KV Cache：

```
请求1: "System: You are a helpful assistant.\nUser: Hello"
请求2: "System: You are a helpful assistant.\nUser: Hi"

Radix Tree:
├── "System: You are a helpful assistant.\nUser: " (共享前缀)
│   ├── "Hello" → 请求1
│   └── "Hi"    → 请求2
```

- **自动前缀复用**：新请求自动匹配最长公共前缀
- **LRU 淘汰**：显存不足时按 LRU 策略淘汰节点
- **零开销**：前缀匹配在 CPU 侧完成，不占用 GPU 计算

### 3.3 效果

- 多轮对话场景：**2-5 倍加速**
- 批量共享前缀推理：**显著提升吞吐**

---

## 4. 大规模部署能力

### 4.1 DeepSeek 大规模部署

SGLang 是 DeepSeek-V3/R1 部署的首选框架之一：

| 部署案例 | 规模 | 效果 |
|----------|------|------|
| DeepSeek + PD + EP | 96x H100 | 大规模专家并行部署 |
| DeepSeek on GB200 NVL72 | 整机柜 | 2.7x Decode 吞吐（Part I） |
| DeepSeek on GB200 NVL72 | 整机柜 | 3.8x Prefill, 4.8x Decode（Part II） |
| GB300 NVL72 | 新一代 | 25x 推理性能提升 |

### 4.2 PD 解耦架构

SGLang 的 Prefill-Decode 解耦支持：
- **异构部署**：Prefill 节点和 Decode 节点独立扩缩容
- **负载均衡**：Cache-Aware Load Balancer
- **高吞吐 + 低延迟**：分别优化两个阶段的矛盾目标

---

## 5. RL 与后训练支持

SGLang 不仅是推理引擎，还是**强化学习（RL）和后训练**的 Rollout 后端：

- **Native RL Integrations**：原生支持 RL 训练流程
- **大规模 Rollout**：高效生成训练数据
- **被采用框架**：AReaL、Miles、slime、Tunix、verl 等

---

## 6. 硬件与模型生态

### 6.1 硬件支持

| 平台 | 支持状态 |
|------|---------|
| NVIDIA GPUs | GB200/B300/H100/A100/Spark/5090 ✅ |
| AMD GPUs | MI355/MI300 ✅ |
| Google TPU | SGLang-Jax 后端 ✅ |
| Intel Xeon CPU | ✅ |
| Huawei Ascend NPU | ✅ |

### 6.2 模型支持

- **语言模型**：Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等
- **Embedding 模型**：e5-mistral、gte、mcdse
- **Reward 模型**：Skywork
- **Diffusion 模型**：WAN、Qwen-Image
- **Diffusion LLM**：LLaDA 2.0

---

> 💡 **核心要点**：SGLang 是**生产级推理的事实标准**之一。RadixAttention 在多轮对话场景的 prefix caching 效果远超传统方案，PD 解耦和大规模 EP 支持使其成为部署 DeepSeek 等 MoE 大模型的首选。此外，其作为 RL Rollout 后端的能力也使其在 AI 训练-推理闭环中占据重要地位。
