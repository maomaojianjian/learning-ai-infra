# PyTorch：深度学习框架

> PyTorch 是 Meta（原 Facebook）开源的 Python 优先深度学习框架，以动态计算图、直观的 API 和强大的生态系统成为大模型训练和推理的事实标准。

---

## 📑 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心组件](#2-核心组件)
- [3. 自动微分与动态图](#3-自动微分与动态图)
- [4. 分布式训练支持](#4-分布式训练支持)
- [5. 编译优化](#5-编译优化)
- [6. 生态系统](#6-生态系统)

---

## 1. 项目概览

| 项目信息 | 内容 |
|---------|------|
| **开发者** | Meta AI + 开源社区 |
| **核心定位** | Python 优先的深度学习研究与生产框架 |
| **开源协议** | BSD 3-Clause |
| **生态地位** | 大模型训练和推理的事实标准框架 |

---

## 2. 核心组件

### 2.1 前端 API

| 模块 | 功能 |
|------|------|
| `torch.Tensor` | 多维张量，支持 CPU/GPU/分布式 |
| `torch.autograd` | 自动微分引擎 |
| `torch.nn` | 神经网络模块和损失函数 |
| `torch.optim` | 优化器（SGD、Adam、AdamW 等） |
| `torch.utils.data` | 数据加载和预处理 |

### 2.2 C++ 后端

| 模块 | 功能 |
|------|------|
| `aten/` | ATen 张量库，底层张量运算 |
| `c10/` | 核心库（Core library），基础设施 |
| `torch/csrc/` | Python 绑定和 C++ 扩展 |

---

## 3. 自动微分与动态图

### 3.1 动态计算图

PyTorch 采用**定义即运行（Define-by-Run）**的急切执行模式：

```python
import torch

x = torch.randn(3, 3, requires_grad=True)
y = x * 2
z = y.sum()
z.backward()  # 自动计算梯度
print(x.grad)  # dz/dx
```

优势：
- **调试友好**：可用标准 Python 调试器
- **控制流灵活**：支持任意 Python 控制流（if/for/while）
- **直觉性强**：代码与数学表达式一一对应

### 3.2 TorchScript

```python
# 将动态模型转换为静态图（用于部署优化）
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

---

## 4. 分布式训练支持

PyTorch 提供原生的分布式训练 API：

| API | 说明 |
|-----|------|
| `torch.nn.DataParallel` | 单进程多卡（已不推荐） |
| `torch.nn.parallel.DistributedDataParallel` | **DDP**，多进程数据并行 |
| `torch.distributed.fsdp` | **FSDP**，Fully Sharded Data Parallel |
| `torch.distributed` | 底层分布式通信原语（基于 NCCL/Gloo/MPI） |
| `torchrun` | 分布式任务启动工具 |

### FSDP 核心思想

- 将模型参数、梯度和优化器状态分片到各 rank
- 前向/反向时按需 all-gather 参数
- 相比 DDP 显著降低显存占用

---

## 5. 编译优化

### 5.1 torch.compile

PyTorch 2.0 引入的编译器：

```python
model = torch.compile(model)
```

- 基于 **TorchDynamo**（图捕获）+ **TorchInductor**（代码生成）
- 自动生成融合 Kernel，减少 Kernel 启动开销
- 与 Triton 深度集成

### 5.2 PyTorch Profiler

```python
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())
```

- 集成 Nsight Systems
- CPU/GPU 时间线分析
- 内存使用分析

---

## 6. 生态系统

PyTorch 拥有最丰富的第三方生态：

| 领域 | 代表库 |
|------|--------|
| **CV** | torchvision、detectron2、segment-anything |
| **NLP** | transformers、tokenizers、accelerate |
| **推理** | vLLM、TensorRT、ONNX Runtime |
| **分布式** | DeepSpeed、Megatron-LM、FairScale |
| **移动端** | PyTorch Mobile、ExecuTorch |

---

> 💡 **核心要点**：PyTorch 是 AI Infra 的**基石框架**。理解其 Autograd 机制、分布式 API（DDP/FSDP）、以及 torch.compile 的优化原理，是进行任何训练/推理工程化的前提。几乎所有主流训练框架和推理引擎都建立在 PyTorch 之上或与其深度集成。
