# 02-CUDA编程与算子优化

> CUDA Kernel 开发、性能优化、经典算子实现、Checkpoint 机制——写出跑得快的 GPU 代码。

---

## 我的笔记

| 文件 | 内容 |
|------|------|
| [LeetCUDA-CUDA学习与实践.md](./LeetCUDA-CUDA学习与实践.md) | LeetCUDA 项目概览、HGEMM 矩阵乘法、FlashAttention 实现、200+ Kernel 分级体系、高性能优化技术全景、Tensor Core 编程层级、Roofline 模型 |
| [Checkpoint机制深度解析.md](./Checkpoint机制深度解析.md) | Checkpoint 核心定义、设计动机、机制分析、训练 vs 推理行为对比、常见策略 |
| [LeetCUDA/](./LeetCUDA/) | LeetCUDA 源码项目（200+ CUDA Kernel + HGEMM + FlashAttention） |

## AIInfraGuide 补充

| 文件 | 内容 |
|------|------|
| [AIInfraGuide/CUDA编程入门指南.md](./AIInfraGuide/CUDA编程入门指南.md) | CUDA 编程入门指南 |
| [AIInfraGuide/第1章-CUDA编程入门.md](./AIInfraGuide/第1章-CUDA编程入门.md) | CUDA 编程入门：环境搭建、编程模型、内存模型 |
| [AIInfraGuide/第2章-CUDA性能优化基础.md](./AIInfraGuide/第2章-CUDA性能优化基础.md) | CUDA 性能优化：Warp、内存访问、Occupancy |
| [AIInfraGuide/第3章-经典算子实现-Reduce.md](./AIInfraGuide/第3章-经典算子实现-Reduce.md) | 经典算子 Reduce：朴素→共享内存→Warp Shuffle |
| [AIInfraGuide/第4章-经典算子实现-GEMM.md](./AIInfraGuide/第4章-经典算子实现-GEMM.md) | 经典算子 GEMM：Tiling、与 cuBLAS 对比 |
| [AIInfraGuide/第5章-经典算子实现-Softmax与算子融合.md](./AIInfraGuide/第5章-经典算子实现-Softmax与算子融合.md) | Softmax 与算子融合：Online Softmax |
| [AIInfraGuide/第6章-Attention算子.md](./AIInfraGuide/第6章-Attention算子.md) | Attention 算子：FlashAttention V1/V2/V3、PagedAttention |
| [AIInfraGuide/第7章-AI编译器.md](./AIInfraGuide/第7章-AI编译器.md) | AI 编译器：Triton、torch.compile、TVM/XLA |
| [AIInfraGuide/第8章-性能分析工具链.md](./AIInfraGuide/第8章-性能分析工具链.md) | 性能分析：Nsight Systems/Compute、PyTorch Profiler |
