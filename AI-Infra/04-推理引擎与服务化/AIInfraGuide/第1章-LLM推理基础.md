---
title: "第1章：LLM 推理基础"
description: "理解 LLM 自回归生成的 Prefill/Decode 两阶段、KV Cache 机制和推理关键性能指标"
pubDate: 2026-04-16
category: "inference-optimization"
order: 30
tags: ["LLM推理", "Prefill", "Decode", "KV Cache", "TTFT", "TPOT"]
---

## 本章简介

推理是大模型从训练走向落地的关键环节。本章建立推理优化的基础概念和分析框架。

**自回归生成过程**详解 LLM 推理的两阶段：Prefill（一次性处理完整 Prompt，Compute Bound）和 Decode（逐 Token 生成，Memory Bound），以及为什么 Decode 阶段效率远低于 Prefill。

**KV Cache**是推理优化的核心概念：缓存已计算的 Key 和 Value 避免重复计算，包括 KV Cache 的生命周期（分配 → 填充 → 使用 → 释放）、显存计算公式和碎片化问题。

**关键性能指标**覆盖 TTFT（首 Token 延迟）、TPOT（每 Token 延迟）、Throughput（吞吐量）、P50/P95/P99 尾延迟和 Goodput（满足 SLO 的有效吞吐）。

**推理链路拆解**走通 Tokenize → Prefill → Decode Loop → Sampling → Detokenize 的完整流程，分析每个环节的计算特性与潜在瓶颈。
