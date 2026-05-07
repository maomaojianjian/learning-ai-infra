# Triton：算子编译器

> Triton 是 OpenAI 开源的 GPU 算子开发语言与编译器，旨在提供比 CUDA 更高的开发效率，同时保持接近手写 Kernel 的性能。它是现代 AI 编译器栈的核心组件。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 设计理念](#2-设计理念)
- [3. 语言特性](#3-语言特性)
- [4. 编译器架构](#4-编译器架构)
- [5. 典型应用](#5-典型应用)
- [6. 硬件支持](#6-硬件支持)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **开发者** | OpenAI / Triton Lang Community |
| **核心定位** | 用于编写高效深度学习算子的语言与编译器 |
| **设计目标** | CUDA 的生产力 + 手写 Kernel 的性能 |
| **开源协议** | MIT |
| **文档** | [triton-lang.org](https://triton-lang.org) |

---

## 2. 设计理念

### 2.1 问题背景

- **CUDA 太难**：需要深入理解线程层级、内存模型、Warp 调度
- **传统 DSL 太局限**：Halide、TVM 等灵活性不足
- **Python 生态**：AI 研究者熟悉 Python，希望用 Python 写算子

### 2.2 Triton 的解法

| 方面 | CUDA | Triton |
|------|------|--------|
| 编程语言 | C++ | Python-like |
| 并行粒度 | 单个线程 | Tile（块） |
| 内存管理 | 手动（Global/Shared/Register）| 自动优化 |
| 线程调度 | 手动 | 编译器自动处理 |
| 开发效率 | 低 | 高 |

---

## 3. 语言特性

### 3.1 Tile-based 编程

Triton 的编程单元是 **Tile（数据块）** 而非单个线程：

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

### 3.2 自动调优（Autotune）

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n_elements']
)
@triton.jit
def kernel(...):
    ...
```

### 3.3 解释器模式

- 无需 GPU 即可调试 Triton Kernel
- 通过 `TRITON_INTERPRET=1` 启用
- 适合开发和 CI 测试

---

## 4. 编译器架构

Triton 基于 **MLIR（Multi-Level Intermediate Representation）**：

```
Python Frontend
      ↓
Triton IR (MLIR Dialect)
      ↓
TritonGPU IR (MLIR Dialect, hardware-agnostic GPU ops)
      ↓
LLVM IR
      ↓
PTX / SASS (NVIDIA) 或 HSACO (AMD)
```

### 4.1 编译流程

1. **Python AST → Triton IR**：捕获 Python 代码的语义
2. **Triton IR → TritonGPU IR**：插入内存布局、同步操作
3. **TritonGPU IR → LLVM IR**： lowered 到硬件指令
4. **LLVM → 机器码**：生成 PTX 或 SASS

---

## 5. 典型应用

### 5.1 FlashAttention 系列

Triton 是实现 FlashAttention 的首选工具：
- FlashAttention-2 Triton 实现
- FlashAttention-3（Hopper 专属优化）

### 5.2 融合算子

Triton 擅长开发融合算子（Fused Kernels）：
- LayerNorm + Linear 融合
- GELU + Linear 融合
- Rotary Embedding 融合

### 5.3 与 PyTorch 集成

```python
# 在 PyTorch 中使用 Triton Kernel
from torch.utils._triton import has_triton

# torch.compile 自动生成 Triton Kernel
model = torch.compile(model)
```

### 5.4 被采用项目

| 项目 | 用途 |
|------|------|
| **PyTorch** | torch.compile 生成 Triton Kernel |
| **vLLM** | 自定义 Attention、MoE Kernels |
| **SGLang** | 高性能算子实现 |
| **TransformerEngine** | FP8 算子 |

---

## 6. 硬件支持

| 平台 | 支持状态 | 最低版本 |
|------|---------|---------|
| **NVIDIA GPUs** | ✅ | SM 8.0+ (Ampere) |
| **AMD GPUs** | ✅ | ROCm 6.2+ |
| **Intel GPUs** | 开发中 | |
| **CPU** | 开发中 | |

---

> 💡 **核心要点**：Triton 是**现代算子开发的革命性工具**。它将 GPU Kernel 开发的抽象层次从"单个线程"提升到"数据块"，配合自动调优和 MLIR 编译器，使得 AI 研究者可以用 Python 的效率写出接近 CUDA 专家手写性能的高性能算子。对于 AI Infra 工程师，掌握 Triton 是进行算子优化的必备技能。
