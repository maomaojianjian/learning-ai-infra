# DeepEP：专家并行通信库

> DeepEP（DeepEveryParallel）是 DeepSeek 开源的高性能 GPU 通信库，专注于 MoE（混合专家模型）的专家并行 all-to-all 通信，提供高吞吐、低延迟的 dispatch/combine Kernel。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心特性](#2-核心特性)
- [3. EPv2 架构升级](#3-epv2-架构升级)
- [4. 性能数据](#4-性能数据)
- [5. 0-SM 实验特性](#5-0-sm-实验特性)
- [6. 使用场景](#6-使用场景)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **全称** | DeepEP (DeepEveryParallel) |
| **开发者** | DeepSeek |
| **核心定位** | MoE 专家并行的高性能 all-to-all GPU 通信库 |
| **编译方式** | 全 JIT（Just-In-Time），安装时无需 CUDA 编译 |
| **后端** | NCCL Gin backend（header-only） |

---

## 2. 核心特性

### 2.1 统一 ElasticBuffer 接口

DeepEP V2 将高吞吐和低延迟 API 统一为单一的 `ElasticBuffer` 接口：
- **高吞吐模式**：适合训练场景的大 batch 通信
- **低延迟模式**：适合推理场景的小 batch 通信
- 自动适配不同规模的专家并行（最高支持 EP2048）

### 2.2 低精度通信支持

- **FP8 dispatching**：专家分发阶段使用 FP8 精度通信
- **BF16 combining**：结果聚合阶段使用 BF16 精度
- 显著降低 MoE 层的通信带宽需求

### 2.3 NCCL Gin Backend

- Header-only 的轻量级通信后端
- 可复用现有 NCCL communicator
- 相比 V1 的 NVSHMEM 后端更轻量、更易集成

---

## 3. EPv2 架构升级

V2 相比 V1 的显著改进：

| 指标 | V1 | V2 | 提升 |
|------|-----|-----|------|
| SM 占用（类似 V3 训练） | 24 SM | 4-6 SM | **4x 节省** |
| 峰值性能 | 基准 | 1.3x | **1.3x 提升** |
| 最大 EP 规模 | 有限 | EP2048 | **规模大幅提升** |
| 自动调参 | 需要 | 解析公式计算 | **无需调参** |

### 3.1 解析式 SM & QP 计算

V2 引入了分析公式直接计算所需的 SM 数量和 Queue Pair 数量，消除了 V1 中耗时的自动调参过程。

### 3.2 混合模式与直接模式

- **Hybrid 模式**：通过 CPU 中介的灵活调度
- **Direct 模式**：GPU 直连的极致性能
- 两种模式在 V2 中均得到保留和优化

---

## 4. 性能数据

测试配置：8K tokens/batch，7168 hidden dim，top-8 experts，FP8 dispatch，BF16 combine

| 架构 | NIC | 拓扑 | Dispatch 带宽 | Combine 带宽 | SM 数 |
|------|-----|------|--------------|-------------|-------|
| SM90 (H100) | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 (H100) | CX7 | EP 8 x 4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 (B100) | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 (B100) | N/A | EP 8 (NVLink) | 726 GB/s | 740 GB/s | 64 |
| SM100 (B100) | N/A | EP 8 (NVLink) | 643 GB/s | 675 GB/s | 24 |

---

## 5. 0-SM 实验特性

DeepEP 还提供一系列实验性的 0-SM（零 SM 占用）特性，利用 RDMA 和 Copy Engine 实现通信与计算完全重叠：

| 特性 | 技术 | 状态 |
|------|------|------|
| **0-SM Engram** | RDMA 远程内存访问 | 实验性 |
| **0-SM PP** | RDMA 流水线并行通信 | 实验性 |
| **0-SM CP** | Copy Engine 上下文并行 | 实验性 |

> 这些特性的目标是：通信完全不占用 GPU 计算单元，实现计算与通信的 100% 重叠。

### 未来规划

- **Elastic GPU & CPU buffers**：连续的虚拟地址空间，底层映射混合的 GPU 和 CPU 物理内存
- **All-gather / Reduce-scatter**：支持 DP & TP 的完整集合通信

---

## 6. 使用场景

| 场景 | 推荐配置 |
|------|---------|
| DeepSeek-V3/R1 训练 | EPv2 + NVLink，4-6 SM |
| DeepSeek-V3/R1 推理 | EPv2 + RDMA，低延迟模式 |
| 其他 MoE 模型 | 根据规模选择 Hybrid/Direct 模式 |

---

> 💡 **核心要点**：DeepEP 是** MoE 专家并行通信**的专用优化库。其核心价值在于：用极少的 SM 资源（4-6 SM）实现接近硬件带宽极限的 all-to-all 通信，同时支持 FP8 低精度通信降低带宽压力。对于部署 DeepSeek 系列模型的 Infra 团队，DeepEP 是不可或缺的组件。
