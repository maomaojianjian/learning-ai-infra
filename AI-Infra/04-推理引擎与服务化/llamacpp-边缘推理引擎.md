# llama.cpp：边缘推理引擎

> llama.cpp 是 ggml-org 维护的纯 C/C++ LLM 推理实现，以零依赖、跨平台、极致量化支持为特点，是边缘设备、本地部署和低成本推理的首选方案。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心特性](#2-核心特性)
- [3. GGUF 格式与量化](#3-gguf-格式与量化)
- [4. 多后端支持](#4-多后端支持)
- [5. 工具与应用](#5-工具与应用)
- [6. 使用场景](#6-使用场景)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **开发者** | ggml-org |
| **核心定位** | 纯 C/C++ LLM 推理，最小依赖，最大兼容性 |
| **开源协议** | MIT |
| **核心库** | [ggml](https://github.com/ggml-org/ggml) — 张量计算后端库 |

---

## 2. 核心特性

### 2.1 零依赖与跨平台

- **纯 C/C++ 实现**，无任何外部依赖
- 支持从嵌入式设备到服务器的全范围硬件

### 2.2 极致量化

llama.cpp 是量化支持最广泛的推理引擎：

| 量化位数 | 说明 |
|----------|------|
| **1.5-bit** | 极端压缩，质量损失较大 |
| **2-bit ~ 8-bit** | 多级整数量化 |
| **Q4_0 / Q4_K_M / Q5_K_M** | 常用高质量量化格式 |
| **IQ 系列** | 重要性感知量化（Imatrix） |
| **MXFP4** | 与 NVIDIA 合作的新格式 |

### 2.3 CPU+GPU 混合推理

- 模型超过显存时，自动将部分层卸载到 CPU 内存
- 支持部分层在 GPU 加速，部分层在 CPU 执行

### 2.4 多模态支持

- `llama-server` 已支持多模态推理（图像+文本）
- 支持 LLaVA 系列等多模态模型

---

## 3. GGUF 格式与量化

### 3.1 GGUF 格式

GGUF（GGML Universal Format）是 llama.cpp 的原生模型格式：
- 单文件包含完整模型权重和元数据
- 支持多种量化方案
- Hugging Face 原生支持（Inference Endpoints、GGUF Editor）

### 3.2 量化流程

```bash
# 从 Hugging Face 模型转换为 GGUF
python convert_hf_to_gguf.py --src MODEL_PATH --dst output.gguf

# 量化
./llama-quantize output.gguf output-q4_k_m.gguf Q4_K_M
```

---

## 4. 多后端支持

| 后端 | 平台 | 优化特性 |
|------|------|---------|
| **Metal** | Apple Silicon | ARM NEON、Accelerate、Metal — 一等公民 |
| **CUDA** | NVIDIA GPUs | 自定义 CUDA Kernels |
| **HIP** | AMD GPUs | ROCm 兼容 |
| **Vulkan** | 跨平台 GPU | 通用 GPU 后端 |
| **SYCL** | Intel GPU | oneAPI |
| **CPU** | x86/ARM/RISC-V | AVX/AVX2/AVX512/AMX、RVV |

---

## 5. 工具与应用

| 工具 | 说明 |
|------|------|
| `llama-cli` | 命令行交互工具 |
| `llama-server` | OpenAI-compatible API 服务器 |
| `llama-bench` | 性能基准测试 |
| `llama.perplexity` | 困惑度评估 |
| `llama.vscode` | VS Code 补全插件 |
| `llama.vim` | Vim/Neovim 补全插件 |

---

## 6. 使用场景

| 场景 | 推荐理由 |
|------|---------|
| **本地/边缘部署** | 零依赖，可运行在无 GPU 设备 |
| **低资源环境** | 1.5-bit ~ 4-bit 量化可将 70B 模型压缩到 20-40GB |
| **Apple Silicon** | Metal 后端优化最佳 |
| **隐私敏感场景** | 完全本地运行，无需联网 |
| **快速原型验证** | 单文件 GGUF，下载即用 |
| **Hugging Face 生态** | 原生支持，与 transformers 无缝衔接 |

---

> 💡 **核心要点**：llama.cpp 的核心价值在于**极致的跨平台兼容性和量化压缩能力**。它证明了通过精细的量化算法和 CPU 优化，大模型可以在消费级硬件上流畅运行。对于需要边缘部署、本地运行或成本敏感的场景，llama.cpp 是不可替代的选择。
