# Megatron-LM：大规模训练框架

> Megatron-LM 是 NVIDIA 开源的 GPU 优化大模型训练库，提供高性能的张量并行、流水线并行和专家并行实现，是业界训练数百亿到数千亿参数模型的标准工具。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心并行策略](#2-核心并行策略)
- [3. Megatron Core 架构](#3-megatron-core-架构)
- [4. 混合精度支持](#4-混合精度支持)
- [5. 性能基准](#5-性能基准)
- [6. 生态工具](#6-生态工具)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **开发者** | NVIDIA |
| **核心定位** | 大规模 Transformer 模型分布式训练框架 |
| **开源协议** | BSD 3-Clause |
| **最新方向** | MoE、Blackwell 支持、Megatron Bridge |

---

## 2. 核心并行策略

Megatron-LM 支持五种并行维度的灵活组合：

| 并行类型 | 英文缩写 | 切分维度 | 适用场景 |
|----------|---------|---------|---------|
| **数据并行** | DP | 数据批次 | 通用 |
| **张量并行** | TP | 层内参数（列/行切分） | 单层参数量大 |
| **流水线并行** | PP | 层间分段 | 模型层数多 |
| **专家并行** | EP | MoE 路由 | 混合专家模型 |
| **上下文并行** | CP | 序列维度 | 长上下文训练 |

### 2.1 张量并行（Tensor Parallelism）

- 对 Attention 和 FFN 层进行列/行切分
- 支持 GQA/MQA 下的 TP 切分优化
- 通信量：每层 2 次 AllReduce（前向 + 反向）

### 2.2 流水线并行（Pipeline Parallelism）

- GPipe / PipeDream / 1F1B 调度策略
- 气泡率分析与优化
- 支持虚拟流水线（Virtual Pipelining）进一步降低气泡

### 2.3 序列并行（Sequence Parallelism）

- Ulysses 风格序列并行
- Ring Attention 变体支持超长序列

---

## 3. Megatron Core 架构

Megatron Core 是 Megatron-LM 的可复用核心库，提供模块化组件：

```
megatron/core/
├── transformer/          # Transformer Block 实现
│   ├── transformer_layer.py
│   ├── attention.py      # 支持 GQA/MQA/MLA
│   ├── mlp.py
│   └── ...
├── parallel_layers/      # 并行层封装
│   ├── tensor_parallel/  # 张量并行层
│   └── pipeline_parallel/# 流水线并行层
├── optimizer/            # 分布式优化器
├── inference/            # 推理支持
├── export/               # 模型导出
└── datasets/             # 数据加载
```

---

## 4. 混合精度支持

| 精度类型 | 支持状态 | 说明 |
|----------|---------|------|
| **FP16** | ✅ | 传统混合精度 |
| **BF16** | ✅ | 更稳定的低精度训练 |
| **FP8** | ✅ | Hopper 架构，Transformer Engine |
| **FP4** | 开发中 | Blackwell 架构 |

---

## 5. 性能基准

Megatron-LM 在 H100 集群上的公开基准：

| 模型规模 | 集群配置 | MFU |
|----------|---------|-----|
| 2B - 462B 参数 | H100 集群 | 最高 47% |

> MFU（Model FLOPs Utilization）是衡量训练效率的关键指标，47% 的 MFU 在业界属于顶尖水平。

---

## 6. 生态工具

### 6.1 Megatron Bridge

- Hugging Face ↔ Megatron 双向 Checkpoint 转换
- 支持模型格式互转，方便预训练与微调衔接

### 6.2 与 DeepSpeed 协作

Megatron-LM 常与 DeepSpeed 配合使用：
- **Megatron-DeepSpeed**：Megatron 的 TP/PP + DeepSpeed 的 ZeRO DP
- 代表模型：MT-530B、BLOOM

---

> 💡 **核心要点**：Megatron-LM 是**张量并行和流水线并行**的业界标杆。理解其 TP 的列/行切分策略、PP 的 1F1B 调度、以及五种并行维度的组合配置，是设计大规模训练架构的核心能力。
