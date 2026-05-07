# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a personal AI Infrastructure learning repository. It contains:

- **`AI-Infra/`** — A 12-module organized knowledge base covering the full AI Infra stack (GPU hardware, CUDA, distributed training, inference engines, K8s scheduling, storage/networking, observability, platform design, course curricula, career growth). Each module contains both study notes (`.md` files) and cloned source code of relevant open-source projects.
- **`AIInfraGuide/`** — An Astro v4 documentation website (source for [caomaolufei.github.io/AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide)). Content is authored in Markdown under `AIInfraGuide/docs/`.
- **`AI_Infra.md`** — The master reference document (2205 lines) from which the `AI-Infra/` knowledge base was derived. When updating either, keep them in sync.
- **`picture/`** — Reference images (WeChat screenshots).
- **`CUDA.pdf`** / **`NVIDIA_Hopper_Architecture_In_Depth.md`** — Reference materials (copied into `AI-Infra/01-前置知识与硬件基础/`).

## AIInfraGuide website

The website is an Astro v4 project with Tailwind CSS, KaTeX math rendering, and Pagefind search.

```bash
cd AIInfraGuide
npm install        # install dependencies
npm run dev        # start dev server at http://localhost:4321
npm run build      # type-check, build, run Pagefind indexing
npm run preview    # preview production build
```

**Key configuration:**
- `astro.config.mjs` — Base path is `/AIInfraGuide`, site URL is `https://caomaolufei.github.io`
- Content lives in `AIInfraGuide/docs/` as Markdown files organized by topic (`guides/`, `prerequisites/`, `cuda/`, `distributed/`, `inference/`, `interview/`)
- The `src/content/` directory configures Astro content collections
- Styling uses Tailwind with the `@tailwindcss/typography` plugin for prose content

## AI-Infra knowledge base structure

The 12 modules under `AI-Infra/` are:

| # | Directory | Focus |
|---|-----------|-------|
| 01 | `01-前置知识与硬件基础/` | GPU/Hopper architecture, CPU/memory, Linux kernel tuning |
| 02 | `02-CUDA编程与算子优化/` | CUDA kernels (LeetCUDA), checkpoint mechanism |
| 03 | `03-分布式训练/` | NCCL, DeepSpeed, Megatron-LM, DeepEP, distributed systems theory |
| 04 | `04-推理引擎与服务化/` | vLLM, SGLang, llama.cpp, serving architecture & benchmarking |
| 05 | `05-AI框架与编译器/` | PyTorch, Triton, AI compiler deep-dive |
| 06 | `06-参考资料/` | Reference summaries, video resources, environment setup |
| 07 | `07-云原生与K8s调度/` | K8s core + AI extensions, cluster scheduling architecture |
| 08 | `08-存储与高速网络/` | Storage tiering (Lustre/MinIO), RDMA/IB/RoCE networking |
| 09 | `09-可观测与成本安全/` | Observability, cost governance, AI security |
| 10 | `10-平台化与架构设计/` | Platform design, architecture diagrams, full-stack maps, deployment pipelines, ecosystem extensions |
| 11 | `11-课程与实战项目/` | Course curricula (CUDA/Triton/vLLM), distributed training algorithms |
| 12 | `12-架构师成长指南/` | Learning plans, production commands, promotion/interview scripts |

Each module directory contains both `.md` study notes and cloned project source code (e.g., `03-分布式训练/DeepSpeed/`, `04-推理引擎与服务化/vllm/`). The `.md` files are the primary authoring surface; the source clones are reference copies for code reading.

## Keeping AI_Infra.md and AI-Infra/ in sync

`AI_Infra.md` is the master document. The `AI-Infra/` directory breaks it into organized per-topic files. When making changes:
- If adding to `AI_Infra.md`, propagate the new content to the relevant `AI-Infra/` module file(s)
- If adding a new `.md` file to `AI-Infra/`, ensure the corresponding section exists in `AI_Infra.md`
- Update `AI-Infra/README.md` (the module index) whenever adding or renaming files
