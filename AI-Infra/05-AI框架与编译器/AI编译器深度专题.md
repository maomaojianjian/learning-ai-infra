# AI 编译器深度专题

> 本模块涵盖 AI 编译器实战项目、MLIR 编译器工具链、模型量化与 NPU 部署等内容。

---

## 📑 目录

- [1. AI 编译器实战项目](#1-ai-编译器实战项目)
- [2. AI 编译器学习路线](#2-ai-编译器学习路线)
- [3. 工业级 Lab 实战](#3-工业级-lab-实战)
- [4. AI 编译器整体架构流程](#4-ai-编译器整体架构流程)
- [5. 编译器技术栈详解](#5-编译器技术栈详解)

---

## 1. AI 编译器实战项目

### 项目一：YOLOv8 目标检测模型的 NPU 移植

- **目标**：实现多路视频流的实时推理
- **应用场景**：安防、智慧交通等领域
- **技术要点**：模型转换、算子适配、性能优化

### 项目二：Llama/Qwen 大语言模型的 NPU 部署

- **涵盖内容**：
  - Attention 算子加速
  - 算子融合优化
  - KV Cache 的 NPU 管理机制
- **最终目标**：部署一个可交互的 NPU 加速聊天机器人

### 核心平台

- 通用 NPU 硬件架构 + MLIR 编译器工具链实践

### 编译路径对比（AOT vs JIT）

| 维度 | AOT 路径（静态链） | JIT 路径（动态链） |
|------|-------------------|-------------------|
| **启动速度** | AOT 胜出 | 较慢 |
| **灵活性** | 固定形状 | JIT 胜出（处理动态形状） |
| **内存开销** | AOT 胜出（零运行时开销） | 有编译延迟 |
| **运行时** | 预编译为可执行二进制 | 运行时轻量编译器生成 Kernel |

---

## 2. AI 编译器学习路线（6 大模块）

| 模块 | 内容 |
|------|------|
| **1. AI 编译器基础** | C/C++、AI 编译器概论、MLIR 基础概念（Dialect/Op/Type）、MLIR Pass 管理与模式匹配 |
| **2. 硬件与环境** | NPU 硬件架构与存储层次、WSL2+Docker 开发环境搭建 |
| **3. MLIR 实战剖析** | 编译器工具链架构与工作流、Frontend（TopDialect）、converter、Backend（TpuDialect 设计）、conversion/lowering、Tpu-mlir 中的 LayerGroup |
| **4. 量化与性能优化** | INT8 量化数学原理与误差分析、校准算法、Profiling 性能分析工具 |
| **5. 模型部署实战（CV）** | YOLOv8 目标检测模型 NPU 部署 |
| **6. LLM 与 Transformer** | Attention 算子部署和优化、KV Cache NPU 管理机制、Qwen3 模型量化与部署 |

---

## 3. 工业级 Lab 实战

| Lab | 内容 |
|-----|------|
| **Lab1** | 手写简单 MLIR 优化 Pass |
| **Lab2** | NPU 基础推理 Demo 运行 |
| **Lab3** | 模型转换流程全跟踪 |
| **Lab4** | 量化精度对比与精度回退 |
| **Lab5** | YOLOv8 Stream 实践 |
| **Lab6** | 部署一个 NPU 加速聊天机器人 |

---

## 4. AI 编译器整体架构流程

```
PyTorch / TensorFlow / JAX / ONNX / Keras
         ↓
    Graph IR（图中间表示）
         ↓
    Multi-level IR（Dialects）
    ├─ 代数优化（Algebraic Opt）
    └─ 内存规划（Memory Planning）
         ↓
    CPU(LLVM) / GPU(NVPTX) / DSA/NPU(Custom ISA)
```

- **编译器前端**：MLIR/HLO Import & Transformation → High-Level IRs → ML Graph Optimizations → Debugging & Intermediate Viz
- **编译器后端**：Low-Level IRs & Dialects → Device-Specific Optimizations → Kernel Library → Executable Binary Generation

---

## 5. 编译器技术栈详解

### 5.1 AI 编译器

| 技术 | 详细介绍 |
|------|---------|
| **CANN** | Compute Architecture for Neural Networks，华为昇腾的 AI 计算架构。提供算子库、编译器、运行时等，对标 NVIDIA CUDA 生态。 |
| **CUDA** | Compute Unified Device Architecture，NVIDIA 的并行计算平台和编程模型。包含驱动、运行时、库（cuBLAS、cuDNN 等），是 GPU 编程的事实标准。 |
| **Triton** | OpenAI 开源的 Python-like GPU 编程语言。简化 CUDA Kernel 开发，让算法工程师也能高效编写自定义算子，被 PyTorch 2.0 集成。 |
| **TVM** | Tensor Virtual Machine，开源深度学习编译器栈。支持多种前端和多后端，自动优化算子性能。 |
| **GLOW** | Facebook 开源的深度学习编译器。基于 LLVM，将神经网络图编译为机器码。 |
| **XLA** | Accelerated Linear Algebra，Google 开发的线性代数编译器。TensorFlow 和 JAX 的后端，通过 JIT 编译优化计算图。 |

### 5.2 传统编译器

| 技术 | 详细介绍 |
|------|---------|
| **LLVM** | Low Level Virtual Machine，模块化编译器基础设施。提供 IR 和优化框架，被广泛用于 AI 编译器的后端代码生成。 |
| **GCC** | GNU Compiler Collection，传统 C/C++ 编译器。在 AI 系统中用于编译底层驱动、通信库等系统软件。 |
| **TC** | Tensor Comprehensions，Facebook 开源的 C++ DSL，用于高效表达张量运算并自动编译优化。 |

### 5.3 模型编译全流程

1. **图优化**：计算图简化、算子融合
2. **常量折叠**：预计算常量
3. **内存规划**：显存分配优化、内存复用
4. **硬件指令生成**：目标代码生成（PTX/SPIRV 等）

---

> 🔑 **核心要点**：AI 编译器是连接算法与硬件的桥梁。MLIR 正在成为编译器基础设施的标准，Triton 让 GPU 算子开发门槛大幅降低，理解编译器从前端到后端的完整流程是 AI Infra 的高级能力。
