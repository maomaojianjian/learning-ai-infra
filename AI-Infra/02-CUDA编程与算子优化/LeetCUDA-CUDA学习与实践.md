# LeetCUDA：CUDA 学习与实践

> LeetCUDA 是一个现代化的 CUDA 学习仓库，包含 200+ 分级 CUDA Kernel、高性能 HGEMM、FlashAttention 实现，以及 100+ 技术博客。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. HGEMM 矩阵乘法优化](#2-hgemm-矩阵乘法优化)
- [3. FlashAttention 实现](#3-flashattention-实现)
- [4. 200+ CUDA Kernel 分级体系](#4-200-cuda-kernel-分级体系)
- [5. Triton 与 CUTLASS 示例](#5-triton-与-cutlass-示例)
- [6. 学习路径建议](#6-学习路径建议)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **全称** | LeetCUDA: Modern CUDA Learn Notes with PyTorch for Beginners |
| **核心内容** | 200+ CUDA Kernels、HGEMM、FlashAttention、100+ Blogs |
| **精度支持** | Tensor/CUDA Cores、TF32/F16/BF16/F8 |
| **性能水平** | HGEMM 可达 cuBLAS 的 98%~100% 性能 |
| **关联项目** | [HGEMM](https://github.com/xlite-dev/HGEMM)、[ffpa-attn](https://github.com/xlite-dev/ffpa-attn.git)、[Cache-DiT](https://github.com/vipshop/cache-dit) |

---

## 2. HGEMM 矩阵乘法优化

HGEMM（Half-precision GEMM）是 CUDA 算子优化的经典练兵场。LeetCUDA 实现了多种 Tensor Core API 的 HGEMM：

| 实现方式 | API | 性能 |
|---------|-----|------|
| WMMA | Warp-level Matrix Multiply Accumulate | ~cuBLAS |
| MMA | 更底层的 Matrix Multiply Accumulate | ~cuBLAS |
| CuTe | CUTLASS 的 CuTe DSL | ~cuBLAS |
| WGMMA | Hopper 架构的 Warp Group MMA | 峰值性能 |

**核心优化技术**：
- Shared Memory Tiling
- 双缓冲（Double Buffering）
- Warp 级数据排布
- 多级流水线隐藏延迟

---

## 3. FlashAttention 实现

LeetCUDA 实现了基于纯 MMA PTX 指令的 FlashAttention-2，包含多种变体：

| 变体 | 特点 |
|------|------|
| **Split KV（Basic, FA-1）** | 基础分块策略 |
| **Split Q（Faster, FA-2）** | 按 Query 分块，更高效的并行 |
| **Split Q + Shared KV** | Shared Memory 复用 KV |
| **Split Q + Shared QKV** | 进一步复用 Q/K/V |
| **Split Q + QK Tiling** | QK 矩阵分块优化 |
| **Split Q + QKV Tiling** | 完整三级分块 |

**关联项目**：
- [ffpa-attn](https://github.com/xlite-dev/ffpa-attn.git)：Faster Flash Prefill Attention，O(1) SRAM 复杂度，支持大 headdim（D=320~1024），比 SDPA EA 快 1.8x~3x

---

## 4. 200+ CUDA Kernel 分级体系

LeetCUDA 的 Kernel 按难度分级，适合循序渐进学习：

### ⭐ Easy / ⭐⭐ Medium
- 基础元素级运算（vector add, elementwise）
- 简单 Reduce 操作
- 矩阵转置、前缀和

### ⭐⭐⭐ Hard
- 高效 Reduce（树形归约、Warp Shuffle）
- GEMM 基础实现
- Softmax 数值稳定实现

### ⭐⭐⭐⭐ Hard+
- Shared Memory Tiling GEMM
- FlashAttention 基础实现
- 多级归约、原子操作优化

### ⭐⭐⭐⭐⭐ Hard++
- 接近 cuBLAS 性能的 HGEMM
- PTX 级优化的 FlashAttention
- Hopper 专属优化（WGMMA、TMA）

### ⭐⭐⭐ Triton / CUTLASS
- Triton Kernel 示例
- CUTLASS/CuTe 算子实现

---

## 5. Triton 与 CUTLASS 示例

LeetCUDA 不仅包含原生 CUDA C++，还涵盖现代算子开发工具链：

- **Triton**：OpenAI 的 Python-like GPU 编程语言，适合快速开发融合算子
- **CUTLASS / CuTe**：NVIDIA 的高性能 CUDA C++ 模板库，用于生产级算子开发

---

## 5+. 高性能 Kernel 优化技术全景

### 关键优化技术

| 技术 | 原理 | 效果 |
|------|------|------|
| **Shared Memory Tiling** | 数据分块到 SMEM，复用减少全局内存访问 | 减少 80%+ HBM 访问 |
| **双缓冲（Double Buffering）** | 两组 SMEM 交替使用，计算和访存重叠 | 隐藏内存延迟 |
| **Warp 级数据排布** | 避免 Bank Conflict，优化 SMEM 访问模式 | SMEM 带宽提升 2-4x |
| **Warp Shuffle** | 寄存器级线程间数据交换（`__shfl_sync`） | 避免 SMEM 来回 |
| **Cooperative Groups** | 灵活线程协作 API，支持跨 Warp/Block 同步 | 更细粒度控制 |
| **多级流水线** | 异步拷贝（TMA/cp.async）+ 计算流水线重叠 | 接近峰值性能 |

### Tensor Core 编程层级

| API | 层级 | 特点 | 适用 |
|-----|------|------|------|
| **WMMA** | Warp 级 | 高层 API，易用性好 | 原型开发 / 简单 GEMM |
| **MMA** | PTX 指令级 | 更底层，灵活控制 | 高性能 GEMM / 复杂计算 |
| **WGMMA** | Hopper Warp Group 级 | 异步执行，峰值性能 | Hopper 架构专属 |
| **CuTe** | CUTLASS DSL | 模板元编程，最大灵活性 | 生产级算子库 |

### Roofline 模型

Roofline 模型是识别 Kernel 瓶颈的核心分析工具：

```
性能 = min(峰值算力, 峰值带宽 × 算术强度)

算术强度 = FLOPs / Bytes（每次内存访问能做多少次浮点运算）

        ↑
   性能  │     _______________ 算力墙（峰值 FLOPS）
        │    /
        │   /
        │  /  带宽墙（斜率 = 峰值带宽）
        │ /
        │/
        └────────────────────→ 算术强度
      计算瓶颈 ←│→ 带宽瓶颈
```

**分析方法**：
1. 计算 Kernel 的算术强度（FLOP / Byte）
2. 与硬件峰值对比：算术强度 < 临界点 → **带宽瓶颈**；算术强度 > 临界点 → **计算瓶颈**
3. 带宽瓶颈优化：算子融合、量化、Shared Memory Tiling
4. 计算瓶颈优化：Tensor Core、混合精度、算法改进

**硬件参考值（H100 SXM）**：
- 峰值算力：~1000 TFLOPS（FP8）、~500 TFLOPS（FP16）
- 峰值带宽：3.35 TB/s（HBM3）
- 临界点：FP16 下 ~150 FLOP/Byte（低于此值为带宽瓶颈）

### 性能分析工具

| 工具 | 层级 | 用途 |
|------|------|------|
| **Nsight Systems** | 系统级 | CPU/GPU 时间线、Kernel 调度、通信重叠 |
| **Nsight Compute** | Kernel 级 | SM 利用率、显存带宽、寄存器压力、Bank Conflict |
| **Nsight Graphics** | 图形级 | 渲染 + 计算结合的调试 |
| **PyTorch Profiler** | 框架级 | 算子耗时、显存使用、与 Nsight 集成 |

---

## 6. 学习路径建议

```
阶段1：基础（1-2周）
  ├── CUDA 编程模型（Thread/Block/Grid）
  ├── 内存管理（Global/Shared/Register）
  └── LeetCUDA Easy/Medium Kernels

阶段2：进阶（2-4周）
  ├── Warp Shuffle、Cooperative Groups
  ├── Shared Memory Tiling、Bank Conflict
  ├── Reduce / GEMM / Softmax 经典算子
  └── LeetCUDA Hard Kernels

阶段3：高级（1-2月）
  ├── Tensor Core（WMMA/MMA）
  ├── FlashAttention 分块策略
  ├── PTX 指令优化
  └── LeetCUDA Hard+/Hard++

阶段4：生产级（持续）
  ├── Triton 算子开发
  ├── CUTLASS/CuTe 模板编程
  ├── Nsight Compute 性能分析
  └── 与 PyTorch 集成自定义算子
```

---

> 💡 **核心要点**：LeetCUDA 的价值在于**分级递进 + 实战导向**。从 Easy Kernel 到 Hard++，配合 HGEMM 和 FlashAttention 的完整实现，是 CUDA 算子开发的最佳入门路径之一。
