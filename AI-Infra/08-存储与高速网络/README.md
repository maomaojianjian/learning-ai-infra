# 08-存储与高速网络

> 三层存储模型、Lustre/MinIO、RDMA/InfiniBand/RoCE 无损网络——大模型训练的基础设施瓶颈。

---

## 我的笔记

| 文件 | 内容 |
|------|------|
| [AI存储体系.md](./AI存储体系.md) | 三层存储模型（热/温/冷）、Lustre 并行文件系统（MDS/OSS/OST/条带化）、MinIO 对象存储（纠删码/多副本/跨区复制）、本地 NVMe 缓存、海量小文件优化、Checkpoint 高频落盘策略、数据版本管理 |
| [AI高速网络架构.md](./AI高速网络架构.md) | 四网物理隔离（管理/存储/训练/业务）、InfiniBand 架构（HCA/SM/分区）、RoCE v2 无损网络（PFC/ECN/DCQCN）、IB vs RoCE 选型对比、NCCL 拓扑感知、网络容灾、故障排查方法论 |
