# 09-可观测与成本安全

> Prometheus/Grafana 监控体系、日志与链路追踪、GPU 成本治理、AI 安全架构。

---

## 我的笔记

| 文件 | 内容 |
|------|------|
| [可观测监控日志追踪.md](./可观测监控日志追踪.md) | 三层指标体系（硬件/集群/业务）、Prometheus/VictoriaMetrics + Grafana、DCGM Exporter、告警体系（降噪/聚合/值班）、Fluentd + Loki 日志系统、OpenTelemetry 链路追踪、混沌工程与 SLO/SLA |
| [成本架构与资源治理.md](./成本架构与资源治理.md) | GPU 成本模型（按需/包年/Spot）、利用率治理（碎片清理/混部/错峰）、存储成本（冷热分层/压缩/生命周期）、多租户计费与成本可视化 |
| [AI安全架构.md](./AI安全架构.md) | 容器安全（Trivy/非root/只读）、RBAC 最小权限、数据脱敏与 TLS 加密、集群安全（VPC/白名单/Falco）、隐私计算（联邦学习/TEE/差分隐私）、对抗防御、模型水印、云边端协同安全 |
