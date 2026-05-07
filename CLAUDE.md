# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a self-contained AI Infrastructure knowledge base. All content lives under `AI-Infra/`.

`AI-Infra/` is organized into 12 modules, each with its own `README.md` serving as a table of contents linking to all files in that module. The 12 modules cover the full AI Infra stack from GPU hardware through distributed training, inference engines, K8s scheduling, storage/networking, observability, platform design, course curricula, and career growth.

Each module contains both `.md` study notes and cloned source code of relevant open-source projects (e.g., `03-分布式训练/DeepSpeed/`, `04-推理引擎与服务化/vllm/`).

## Module index

| # | Directory | Focus |
|---|-----------|-------|
| 01 | `01-前置知识与硬件基础/` | GPU/Hopper architecture, CPU/memory, Linux kernel tuning |
| 02 | `02-CUDA编程与算子优化/` | CUDA kernels (LeetCUDA), checkpoint mechanism |
| 03 | `03-分布式训练/` | NCCL, DeepSpeed, Megatron-LM, DeepEP, distributed systems theory |
| 04 | `04-推理引擎与服务化/` | vLLM, SGLang, llama.cpp, serving architecture & benchmarking |
| 05 | `05-AI框架与编译器/` | PyTorch, Triton, AI compiler deep-dive |
| 06 | `06-参考资料/` | Reference summaries, video resources, environment setup, interview questions |
| 07 | `07-云原生与K8s调度/` | K8s core + AI extensions, cluster scheduling architecture |
| 08 | `08-存储与高速网络/` | Storage tiering (Lustre/MinIO), RDMA/IB/RoCE networking |
| 09 | `09-可观测与成本安全/` | Observability, cost governance, AI security |
| 10 | `10-平台化与架构设计/` | Platform design, architecture diagrams, full-stack maps, ecosystem extensions |
| 11 | `11-课程与实战项目/` | Course curricula (CUDA/Triton/vLLM), distributed training algorithms |
| 12 | `12-架构师成长指南/` | Learning plans, production commands, promotion/interview scripts |

## Key files

- `AI-Infra/README.md` — Master index of all 12 modules
- `AI-Infra/CLAUDE.md` — Claude Code guidance for working inside AI-Infra/
- Each module's `README.md` — Module-specific TOC linking to all files within
