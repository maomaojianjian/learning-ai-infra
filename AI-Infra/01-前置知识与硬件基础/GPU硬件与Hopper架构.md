# GPU 硬件与 Hopper 架构

> 本模块汇总当前目录下的 GPU 硬件相关资料，包括 NVIDIA Hopper 架构深度解析、Hopper 架构图片集、以及 CUDA 编程基础。

---

## 📑 目录

- [1. NVIDIA GPU 架构演进](#1-nvidia-gpu-架构演进)
- [2. Hopper 架构核心特性](#2-hopper-架构核心特性)
- [3. H100 GPU 规格详解](#3-h100-gpu-规格详解)
- [4. 互联技术](#4-互联技术)
- [5. 显存与带宽](#5-显存与带宽)
- [6. CUDA 编程模型基础](#6-cuda-编程模型基础)
- [7. 参考资料清单](#7-参考资料清单)

---

## 1. NVIDIA GPU 架构演进

| 架构 | 年份 | 代表产品 | 关键特性 |
|------|------|----------|----------|
| **Volta** | 2017 | V100 | 首次引入 Tensor Core |
| **Ampere** | 2020 | A100 | TF32/BF16 支持，MIG 多实例，第二代 Tensor Core |
| **Hopper** | 2022 | H100 | FP8 支持，Transformer Engine，第四代 Tensor Core |
| **Blackwell** | 2024 | B200 | 第二代 Transformer Engine，FP4 支持 |

---

## 2. Hopper 架构核心特性

基于 `NVIDIA_Hopper_Architecture_In_Depth.md` 整理：

### 2.1 全新流式多处理器（SM）

- **第四代 Tensor Core**：相比 A100，芯片到芯片速度提升高达 6 倍；FP8 数据类型下是 A100 的 4 倍
- **全新 DPX 指令**：动态规划算法加速（如 Smith-Waterman、Floyd-Warshall），比 A100 快 7 倍
- **IEEE FP64/FP32 处理速率提升 3 倍**
- **线程块集群（Thread Block Cluster）**：CUDA 编程模型新增层次，支持跨多个 SM 的线程同步与协作
- **分布式共享内存（Distributed Shared Memory）**：支持跨 SM 直接进行 SM-to-SM 通信

### 2.2 异步执行特性

- **张量内存加速器（TMA, Tensor Memory Accelerator）**：高效传输全局内存与共享内存间的大块数据
- **异步事务屏障（Asynchronous Transaction Barrier）**：支持原子数据移动与同步
- H100 是首款真正的**异步 GPU**，可构建端到端异步流水线，完全重叠数据移动与计算

### 2.3 Transformer Engine

结合软件与 Hopper Tensor Core 技术，智能管理 FP8 和 16 位计算的动态选择：
- 大型语言模型 **AI 训练速度提升高达 9 倍**
- **AI 推理速度提升高达 30 倍**

### 2.4 第二代 MIG（多实例 GPU）

- 每个 GPU 实例计算容量提升约 3 倍
- 内存带宽提升近 2 倍
- 最多支持 7 个独立 GPU 实例
- 首次在 MIG 级别提供机密计算能力

---

## 2+. GPU 虚拟化技术全览

### MIG 硬隔离

- **原理**：物理切分 GPU（SM + 显存），每个实例独立故障域
- **隔离级别**：算力硬隔离 + 显存硬隔离
- **适用**：多租户生产环境，需严格 QoS 保障
- **限制**：实例数有限（H100 最多 7 个），切分配置固定

### MPS 软共享（Multi-Process Service）

- **原理**：多进程共享同一 GPU，MPS Server 统一调度 Kernel
- **优势**：灵活分配、无需重启、利用率高
- **风险**：
  - 显存抢占：进程 A 可能耗尽显存影响进程 B
  - QoS 难以保证：算力竞争导致延迟抖动
  - 调度冲突：多进程 Kernel 争抢 SM
  - 错误隔离弱：单进程崩溃可能影响其他进程
- **适用**：微调/小模型推理等非核心任务

### vGPU（Virtual GPU）

- **原理**：时间片切分 GPU 给多个虚拟机
- **显存 QoS**：固定显存分配，防止越界
- **算力限制**：时间片调度，软限制
- **适用**：虚拟化环境、云平台多租户

### 三种方案对比

| | MIG | MPS | vGPU |
|--|-----|-----|------|
| **隔离级别** | 硬件级 | 进程级 | 虚拟机级 |
| **显存隔离** | 硬隔离 | 无隔离 | 固定分配 |
| **算力隔离** | 硬隔离 | 竞争 | 时间片 |
| **灵活性** | 低 | 高 | 中 |
| **适用场景** | 多租户生产 | 微调/小模型 | 云平台 |

---

## 2++. 异构算力生态

### 主流 AI 芯片

| 厂商 | 芯片 | 架构特点 | 软件栈 |
|------|------|---------|--------|
| **NVIDIA** | H100/H200/B200 | CUDA 成熟生态 | CUDA/cuBLAS/cuDNN/NCCL |
| **AMD** | MI300X/MI250X | CDNA3 架构，192GB HBM3 | ROCm/HIP/MIOpen/RCCL |
| **华为昇腾** | 910B/910C | Da Vinci 架构，HCCS 互联 | CANN/MindSpore/HCCL |
| **昆仑芯** | R300 | 自研 XPU 架构，PCIe 5.0 | XPU SDK/XTCL |
| **摩尔线程** | MTT S4000 | MUSA 架构，兼容 CUDA | MUSIFY（CUDA 自动转换） |

### 统一纳管难点

- **驱动生态碎片化**：每家自研驱动 + 通信库，API 不兼容
- **Device Plugin 差异**：K8s 设备插件需分别开发和适配
- **通信库异构**：NCCL（NVIDIA）/ RCCL（AMD）/ HCCL（昇腾），集合通信实现不同
- **精度对齐**：不同芯片 FP16/BF16/FP8 表现不同，需逐模型验证
- **性能调优**：各家算子库、融合策略、内存模型均有差异

### 纳管策略

1. **统一设备抽象层**：封装各芯片驱动，对上层暴露统一接口
2. **异构调度**：K8s 多 Device Plugin，按芯片类型分别调度
3. **模型多后端编译**：TVM/Triton 多硬件后端，一次编写多芯片部署
4. **性能基线**：每芯片建立性能基准，量化精度对齐标准

---

## 3. H100 GPU 规格详解

### 3.1 完整 GH100 GPU

| 规格项 | 数值 |
|--------|------|
| 制程工艺 | 台积电 4N 定制工艺 |
| 晶体管数量 | 800 亿 |
| 裸片尺寸 | 814 mm² |
| GPC 数量 | 8 个 |
| TPC 数量 | 72 个（每个 GPC 9 个） |
| SM 数量 | 144 个 |
| FP32 CUDA Core | 18,432 个（每 SM 128 个） |
| 第四代 Tensor Core | 576 个（每 SM 4 个） |
| HBM3 / HBM2e 堆栈 | 6 个 |
| 内存控制器 | 12 个（512 位） |
| L2 缓存 | 60 MB |

### 3.2 SXM5 版 H100 GPU

| 规格项 | 数值 |
|--------|------|
| SM 数量 | 132 个 |
| FP32 CUDA Core | 16,896 个 |
| Tensor Core | 528 个 |
| HBM3 容量 | 80 GB |
| HBM3 带宽 | 3.35 TB/s |
| L2 缓存 | 50 MB |
| NVLink 带宽 | 900 GB/s（双向） |
| PCIe 版本 | Gen 5（128 GB/s 总带宽） |

---

## 4. 互联技术

| 技术 | 带宽 | 用途 |
|------|------|------|
| **NVLink 4.0** | 900 GB/s（双向） | GPU 之间直连 |
| **NVSwitch 3.0** | 13.6 Tbits/s | 节点内全对全互联 |
| **NVLink Switch System** | 57.6 TB/s（256 GPU 全对全） | 跨节点 GPU 互联，最多 256 GPU |
| **InfiniBand NDR** | 400 Gbps | 跨节点网络 |
| **PCIe Gen 5** | 128 GB/s（双向 64 GB/s） | CPU-GPU、外设连接 |

---

## 5. 显存与带宽

### 5.1 HBM（高带宽内存）

- **H100 SXM**：80GB HBM3，带宽 3.35 TB/s
- **A100 SXM**：80GB HBM2e，带宽 2.0 TB/s

### 5.2 显存墙问题

很多 AI 工作负载（尤其是推理）是**内存带宽瓶颈**而非计算瓶颈。这意味着：
- 显存带宽提升有时比算力提升更重要
- 优化策略：量化降低显存占用、算子融合减少访存、KV Cache 管理

---

## 6. CUDA 编程模型基础

### 6.1 线程层级

```cpp
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 启动 Kernel
int blocks = (n + 255) / 256;
vectorAdd<<<blocks, 256>>>(d_a, d_b, d_c, n);
```

| 层级 | 说明 |
|------|------|
| **Thread** | 最小执行单元 |
| **Block** | 一组线程，共享 Shared Memory |
| **Grid** | 所有 Block 的集合 |
| **Cluster（Hopper 新增）** | 保证并发调度的线程块组，支持跨 SM 协作 |

### 6.2 内存层级

| 内存类型 | 速度 | 容量 | 生命周期 | 可见性 |
|----------|------|------|----------|--------|
| 寄存器 | 最快 | 有限 | Thread | 单线程 |
| Shared Memory | 快 | 每 Block 有限 | Block | Block 内 |
| L1/L2 Cache | 较快 | 自动管理 | 自动 | 所有线程 |
| 全局内存（HBM）| 慢 | 大 | 手动 | 所有线程 |

---

## 7. 参考资料清单

| 资料名称 | 路径 | 说明 |
|---------|------|------|
| NVIDIA Hopper 架构深度解析 | `../NVIDIA_Hopper_Architecture_In_Depth.md` | H100 架构详细技术文章 |
| Hopper 架构图片集 | `../hopper_images/` | 包含 A100 vs H100 对比、TMA、Cluster 等图片 |
| CUDA 编程指南（PDF） | `../CUDA.pdf` | NVIDIA 官方 CUDA 编程手册 |

---

> 🔑 **核心要点**：Hopper 架构的最大创新是**异步执行**和**Transformer Engine**。理解 Tensor Core、NVLink、HBM 带宽这三者对后续分布式训练和推理优化至关重要。
