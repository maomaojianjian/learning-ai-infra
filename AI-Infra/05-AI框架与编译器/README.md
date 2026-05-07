# 05-AI框架与编译器

> PyTorch 深度学习框架、Triton 算子编译器、AI 编译器专题——连接算法与硬件的桥梁。

---

## 我的笔记

| 文件 | 内容 |
|------|------|
| [PyTorch-深度学习框架.md](./PyTorch-深度学习框架.md) | 动态计算图、Autograd、TorchScript、torch.compile、DDP/FSDP、C++ 后端（ATen/C10）、PyTorch Profiler |
| [Triton-算子编译器.md](./Triton-算子编译器.md) | Tile-based 编程模型、与 CUDA 对比、Autotune、MLIR 编译器架构（Triton IR→LLVM IR→PTX）、硬件支持 |
| [AI编译器深度专题.md](./AI编译器深度专题.md) | NPU 移植项目（YOLOv8/Llama）、MLIR 学习路线、6 大 Lab 实战、编译器全流程（图优化→常量折叠→内存规划→代码生成）、CANN/TVM/XLA/GLOW 技术栈 |
| [pytorch/](./pytorch/) | PyTorch 源码 |
| [triton/](./triton/) | Triton 源码 |

## AIInfraGuide 补充

| 文件 | 内容 |
|------|------|
| [AIInfraGuide/PyTorch框架入门.md](./AIInfraGuide/PyTorch框架入门.md) | PyTorch 框架入门 |
| [AIInfraGuide/第4章-PyTorch框架.md](./AIInfraGuide/第4章-PyTorch框架.md) | PyTorch 框架：Tensor、自动微分、训练流程 |
