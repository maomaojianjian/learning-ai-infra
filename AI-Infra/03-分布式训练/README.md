# 03-分布式训练

> 分布式系统理论、NCCL 通信、DeepSpeed/Megatron/DeepEP 训练框架——大模型训练的通信骨架。

---

## 我的笔记

| 文件 | 内容 |
|------|------|
| [分布式系统理论基础.md](./分布式系统理论基础.md) | CAP/BASE 理论、Raft/Paxos 共识、分布式锁、流量控制、负载均衡、AI 场景专属约束 |
| [NCCL-GPU通信基础.md](./NCCL-GPU通信基础.md) | NCCL 通信原语、Ring/Tree 算法、互联拓扑、调优参数、Gloo/MPI 对比、torchrun/mpi4py 训练环境标准化 |
| [DeepSpeed-分布式训练优化.md](./DeepSpeed-分布式训练优化.md) | ZeRO-1/2/3 系列、3D 并行、Ulysses 序列并行、DeepSpeed-MoE、SuperOffload/ZenFlow 等前沿创新 |
| [Megatron-LM-大规模训练框架.md](./Megatron-LM-大规模训练框架.md) | TP/PP/DP/EP/CP 五种并行、Megatron Core 架构、混合精度、Megatron Bridge |
| [DeepEP-专家并行通信库.md](./DeepEP-专家并行通信库.md) | MoE AllToAll 通信、EPv2 架构（4x SM节省）、0-SM 实验特性、性能数据 |
| [DeepSpeed/](./DeepSpeed/) | DeepSpeed 源码 |
| [Megatron-LM/](./Megatron-LM/) | Megatron-LM 源码 |
| [DeepEP/](./DeepEP/) | DeepEP 源码 |
| [nccl/](./nccl/) | NCCL 源码 |

## AIInfraGuide 补充

| 文件 | 内容 |
|------|------|
| [AIInfraGuide-communication/collective-communication-primer.md](./AIInfraGuide-communication/collective-communication-primer.md) | 集群通信网络与 NCCL |
| [AIInfraGuide-communication/第6章-集合通信基础.md](./AIInfraGuide-communication/第6章-集合通信基础.md) | 集合通信基础 |
| [AIInfraGuide-分布式训练/第1章-分布式训练总论.md](./AIInfraGuide-分布式训练/第1章-分布式训练总论.md) | 分布式训练总论：显存分析、并行策略全景 |
| [AIInfraGuide-分布式训练/第2章-数据并行.md](./AIInfraGuide-分布式训练/第2章-数据并行.md) | DP/DDP/FSDP |
| [AIInfraGuide-分布式训练/第3章-ZeRO系列.md](./AIInfraGuide-分布式训练/第3章-ZeRO系列.md) | ZeRO-1/2/3、ZeRO-Offload |
| [AIInfraGuide-分布式训练/第4章-张量并行与序列并行.md](./AIInfraGuide-分布式训练/第4章-张量并行与序列并行.md) | TP/SP、GQA/MQA 切分 |
| [AIInfraGuide-分布式训练/第5章-流水线并行.md](./AIInfraGuide-分布式训练/第5章-流水线并行.md) | GPipe/PipeDream/1F1B、气泡率分析 |
| [AIInfraGuide-分布式训练/第6章-3D并行与混合训练策略.md](./AIInfraGuide-分布式训练/第6章-3D并行与混合训练策略.md) | 3D 并行组合、混合精度、梯度累积、Activation Checkpointing |
| [AIInfraGuide-分布式训练/第7章-训练框架实战.md](./AIInfraGuide-分布式训练/第7章-训练框架实战.md) | Megatron-LM、DeepSpeed 实战配置 |
| [AIInfraGuide-分布式训练/pytorch-distributed.md](./AIInfraGuide-分布式训练/pytorch-distributed.md) | PyTorch 分布式训练：从原理到实战 |
