# 04-推理引擎与服务化

> vLLM、SGLang、llama.cpp 推理引擎、服务化架构、性能压测——LLM 推理全链路。

---

## 我的笔记

| 文件 | 内容 |
|------|------|
| [vLLM-高效LLM推理服务.md](./vLLM-高效LLM推理服务.md) | PagedAttention、Continuous Batching、量化方案、推理引擎全景对比、引擎架构基础 |
| [SGLang-高性能推理框架.md](./SGLang-高性能推理框架.md) | RadixAttention、零开销 CPU 调度、PD 解耦、大规模 EP 部署、RL 后训练支持 |
| [llamacpp-边缘推理引擎.md](./llamacpp-边缘推理引擎.md) | GGUF 格式、极致量化（IQ系列）、CPU+GPU 混合推理、多后端支持 |
| [推理服务化架构与压测.md](./推理服务化架构与压测.md) | 网关→调度→模型后端分层架构、多模型混部、动态加载、限流熔断灰度、性能压测方法论、稀疏推理/蒸馏/动态限流 |
| [vllm/](./vllm/) | vLLM 源码 |
| [sglang/](./sglang/) | SGLang 源码 |
| [llama.cpp/](./llama.cpp/) | llama.cpp 源码 |

## AIInfraGuide 补充

| 文件 | 内容 |
|------|------|
| [AIInfraGuide/第1章-LLM推理基础.md](./AIInfraGuide/第1章-LLM推理基础.md) | LLM 推理基础：Prefill/Decode、KV Cache、性能指标 |
| [AIInfraGuide/第2章-推理引擎核心技术.md](./AIInfraGuide/第2章-推理引擎核心技术.md) | PagedAttention、Continuous Batching、调度策略 |
| [AIInfraGuide/第3章-主流推理框架/第3章-主流推理框架.md](./AIInfraGuide/第3章-主流推理框架/第3章-主流推理框架.md) | 主流推理框架概览 |
| [AIInfraGuide/第3章-主流推理框架/vllm快速入门.md](./AIInfraGuide/第3章-主流推理框架/vllm快速入门.md) | vLLM 快速入门 |
| [AIInfraGuide/第4章-量化.md](./AIInfraGuide/第4章-量化.md) | INT8/INT4、GPTQ、AWQ、SmoothQuant |
| [AIInfraGuide/第5章-Speculative-Decoding.md](./AIInfraGuide/第5章-Speculative-Decoding.md) | 投机解码原理、Draft Model、验证策略 |
| [AIInfraGuide/第6章-PD解耦架构.md](./AIInfraGuide/第6章-PD解耦架构.md) | Prefill-Decode 解耦部署、异构推理 |
| [AIInfraGuide/第7章-性能分析与Benchmark.md](./AIInfraGuide/第7章-性能分析与Benchmark.md) | 推理性能评估、延迟/吞吐量分析 |
| [AIInfraGuide/第8章-推理优化选型与端到端实战.md](./AIInfraGuide/第8章-推理优化选型与端到端实战.md) | 方案选型、部署实战、生产最佳实践 |
