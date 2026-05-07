---
title: "第3章：ZeRO 系列（DeepSpeed）"
description: "深入理解 ZeRO-1/2/3 的切分策略、通信量分析，以及 ZeRO-Offload/Infinity 的 CPU/NVMe 卸载机制"
pubDate: 2026-04-16
category: "distributed-training"
order: 22
tags: ["ZeRO", "DeepSpeed", "显存优化", "Offload"]
---

## 本章简介

ZeRO（Zero Redundancy Optimizer）是 DeepSpeed 的核心技术，通过逐步切分训练状态来突破单卡显存限制。本章逐层拆解 ZeRO 的三个阶段。

**ZeRO 核心思想**：训练状态由优化器状态、梯度和参数三部分组成，"切分-聚合"范式使每卡只存 1/N，需要时通信获取。

**ZeRO-1**切分优化器状态，**ZeRO-2**进一步切分梯度（backward 时 ReduceScatter），**ZeRO-3**连参数也切分（forward/backward 都需 AllGather，用完即弃）。每个阶段都配有通信模式分析和显存节省计算。

**ZeRO-Offload / ZeRO-Infinity**将优化器状态卸载到 CPU 内存甚至 NVMe SSD，适用于单卡或少卡训练超大模型的场景。

**ZeRO 选型指南**提供 ZeRO-1/2/3 的适用场景对比和通信量 vs 显存节省的 trade-off 表。
