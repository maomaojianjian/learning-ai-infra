---
title: "第5章：Speculative Decoding"
description: "理解投机解码的核心原理（Draft + Verify）、Self-Draft 方案（Medusa/EAGLE-2）及其收益边界与限制"
pubDate: 2026-04-16
category: "inference-optimization"
order: 34
tags: ["Speculative Decoding", "投机解码", "Medusa", "EAGLE", "Rejection Sampling"]
---

## 本章简介

自回归解码的串行瓶颈是 Decode 阶段效率低的根本原因。Speculative Decoding 通过"先猜后验"打破这一瓶颈。

**核心原理**详解 Speculative Sampling 框架：Draft 模型快速猜测多个 Token，Target 模型并行验证。通过 Rejection Sampling 机制保证正确性——数学上证明接受的 Token 严格服从 Target 分布。

**Draft 模型方案**讨论独立小模型的选择（大小、架构、能力匹配）以及 Acceptance Rate 对加速比的影响。

**Self-Draft 方案**覆盖 Medusa（多 Decoding Head 预测多 Token）和 EAGLE-2（动态 Draft Tree + 置信度校准），以及从 Token 级到 Block 级联合验证的演进。

**收益边界与限制**分析高接受率场景（代码生成，加速明显）vs 低接受率场景（开放对话，收益有限），以及与量化叠加的精度风险、与 Continuous Batching 的调度复杂度。

**动手实验**：用 vLLM 对比代码生成 vs 开放对话的 Acceptance Rate 差异。
