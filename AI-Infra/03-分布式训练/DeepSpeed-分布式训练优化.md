# DeepSpeed：分布式训练优化

> DeepSpeed 是微软开源的深度学习优化库，通过 ZeRO、3D 并行、序列并行等创新，实现了极致规模的大模型训练。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心技术体系](#2-核心技术体系)
- [3. ZeRO 系列详解](#3-zero-系列详解)
- [4. 并行策略](#4-并行策略)
- [5. 近期创新](#5-近期创新)
- [6. 典型应用](#6-典型应用)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **开发者** | Microsoft DeepSpeed Team |
| **核心定位** | 极致规模的深度学习训练优化库 |
| **代表性模型** | MT-530B、BLOOM、Jurassic-1、GLM-130B |
| **开源协议** | Apache 2.0 |
| **社区活跃度** | 高，持续更新（最新 SuperOffload、ZenFlow 等） |

---

## 2. 核心技术体系

| 技术 | 说明 |
|------|------|
| **ZeRO** | Zero Redundancy Optimizer，消除数据并行中的显存冗余 |
| **ZeRO-Infinity** | 将优化器状态卸载到 CPU/NVMe，支持万亿参数模型 |
| **ZeRO-Offload** | CPU 卸载引擎，单 GPU 训练大模型 |
| **3D 并行** | 数据并行 + 张量并行 + 流水线并行的组合 |
| **Ulysses 序列并行** | 长序列训练的序列并行策略 |
| **DeepSpeed-MoE** | 混合专家模型（Mixture-of-Experts）训练与推理 |
| **DeepSpeed-Inference** | 推理优化，包含压缩和量化库 |
| **DeepCompile** | 分布式训练的编译器优化 |

---

## 3. ZeRO 系列详解

### 3.1 ZeRO 核心思想

传统数据并行（DDP）中，每张 GPU 都保存完整的模型参数、梯度和优化器状态，造成巨大的显存冗余。

ZeRO 通过**分片（Shard）**消除冗余：

| ZeRO 阶段 | 分片内容 | 显存节省 |
|-----------|---------|---------|
| **ZeRO-1** | 优化器状态分片 | 4x |
| **ZeRO-2** | 优化器状态 + 梯度分片 | 8x |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数分片 | 与数据并行度线性相关 |

### 3.2 ZeRO-Offload / ZeRO-Infinity

- **Offload**：将优化器状态和部分计算卸载到 CPU 内存
- **Infinity**：进一步卸载到 NVMe SSD，支持训练万亿参数模型
- **SuperOffload**：最新成果，大规模 LLM 在 Superchip 上的卸载优化
- **ZenFlow**：无停顿的 Offload 引擎

---

## 4. 并行策略

### 4.1 3D 并行

```
数据并行（DP）    →  多张 GPU 各自处理不同数据批次
张量并行（TP）    →  单层参数切分到多张 GPU
流水线并行（PP）  →  模型按层分段，数据流水式通过
```

DeepSpeed 提供灵活的并行组合配置，自动优化通信拓扑。

### 4.2 Ulysses 序列并行

针对长上下文（Long Context）训练的序列并行策略：
- 将序列维度切分到多个 GPU
- 配合 DeepSpeed 的通信优化，支持百万级 Token 序列训练
- **Ulysses-Offload**： democratize 长上下文 LLM 训练

### 4.3 DeepSpeed-MoE

- 支持混合专家模型的训练与推理
- 专家并行的路由优化
- 与 ZeRO 结合，实现大规模 MoE 模型训练

---

## 5. 近期创新

| 时间 | 技术 | 说明 |
|------|------|------|
| 2026/03 | SuperOffload | 获 ASPLOS 2026 最佳论文荣誉提名 |
| 2025/10 | ZenFlow | 无停顿 Offload 引擎 |
| 2025/06 | ALST | Arctic 长序列训练 |
| 2025/04 | DeepCompile | 分布式训练编译器优化 |
| 2024/12 | Ulysses-Offload | 长上下文训练民主化 |

---

## 6. 典型应用

DeepSpeed 已被用于训练多个超大规模模型：

| 模型 | 参数量 | 说明 |
|------|--------|------|
| Megatron-Turing NLG | 530B | 当时世界最大语言模型 |
| BLOOM | 176B | 多语言大模型 |
| Jurassic-1 | 178B | AI21 Labs |
| GLM-130B | 130B | 清华智谱 |
| xTrimoPGLM | 100B | 蛋白质语言模型 |

---

> 💡 **核心要点**：DeepSpeed 的最大价值在于 **ZeRO 系列优化**，它让研究人员可以用更少的 GPU 训练更大的模型。对于 AI Infra 工程师，理解 ZeRO 的分片原理、通信模式和 Offload 机制是分布式训练优化的必修课。
