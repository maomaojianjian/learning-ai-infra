---
title: "第6章：Prefill/Decode 解耦架构"
description: "理解 P/D 混合 Batching 的问题、DistServe/Splitwise 等解耦方案、Goodput 与 SLO 感知调度"
pubDate: 2026-04-16
category: "inference-optimization"
order: 35
tags: ["P/D解耦", "DistServe", "Splitwise", "Goodput", "SLO"]
---

## 本章简介

Prefill 和 Decode 的计算特性截然不同（Compute Bound vs Memory Bound），混在一起会互相干扰。本章深入解耦架构的设计与挑战。

**混合 Batching 的问题**定量分析 Prefill 对 Decode 的干扰：Decode P95 TPOT 可被拖慢 3-5 倍。

**解耦架构设计**覆盖 DistServe（OSDI'24，系统化论证 P/D 解耦）、Splitwise（ISCA'24，分配到不同 GPU 池）、TaiChi（聚合与解耦统一框架）和 MLC Microserving（跨引擎编排）。

**Goodput 与 SLO 感知调度**强调 Raw QPS 不等于用户体验，Goodput（满足 SLO 的有效吞吐）才是真正的优化目标。

**解耦架构的挑战**包括 KV Cache 从 Prefill 节点迁移到 Decode 节点的带宽压力、调度器复杂度、P/D GPU 池的资源配比推导。

**动手实验**：构造混合负载场景量化 P/D 互扰程度，推导给定工作负载下的 P/D GPU 池配比。
