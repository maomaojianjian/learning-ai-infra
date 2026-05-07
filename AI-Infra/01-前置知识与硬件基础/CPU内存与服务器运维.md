# CPU & 内存体系与服务器运维

> 本模块涵盖 AI 服务器的 CPU 架构、内存技术、总线拓扑及硬件运维，是架构设计的底层基础。

---

## 📑 目录

- [1. CPU 架构](#1-cpu-架构)
- [2. 内存技术](#2-内存技术)
- [3. 总线拓扑](#3-总线拓扑)
- [4. AI 服务器整机架构](#4-ai-服务器整机架构)
- [5. 硬件故障域设计](#5-硬件故障域设计)
- [6. Linux 高性能内核](#6-linux-高性能内核)

---

## 1. CPU 架构

### 1.1 x86/ARM 差异

- **x86**：大核/小核调度、CPU 睿频/功耗限制
- **ARM**：能效比高，部分推理场景适用
- **NUMA 架构**：多 NUMA 节点、跨 NUMA 访问延迟、本地内存/远端内存性能差
- **CPU-GPU 亲和拓扑**：CPU 直连 GPU vs 南桥转接 GPU 性能差异

### 1.2 NUMA 详解

NUMA（Non-Uniform Memory Access）是多路服务器的核心架构：

| 访问类型 | 延迟 | 带宽 |
|----------|------|------|
| 本地内存 | 低 | 高 |
| 远端内存 | 高（1.5-2x） | 低 |

关键排查命令：
```bash
numastat          # NUMA 统计
lscpu -e          # CPU 拓扑
numactl --hardware # NUMA 硬件拓扑
```

---

## 2. 内存技术

### 2.1 DDR4/DDR5

- DDR5 带宽更高、功耗更低
- AI 训练场景内存带宽同样关键

### 2.2 大页内存（HugePage）

```bash
# 预留大页
echo 524288 > /proc/sys/vm/nr_hugepages

# 透明大页 THP — 自动管理
# /sys/kernel/mm/transparent_hugepage/enabled
```

### 2.3 锁内存 mlock

防止内存交换，保障 AI 大内存进程：
- 训练任务关键内存锁定
- 避免 OOM killer 误杀

---

## 3. 总线拓扑

- 多卡拓扑冲突排查
- 硬件 BMC 管理：风扇/功耗/温度风控
- PCIe 通道分配与 NUMA 绑核

---

## 4. AI 服务器整机架构

| 机型 | GPU 配比 | 适用场景 |
|------|----------|----------|
| **训练机型** | 高 GPU 配比（8×H100 SXM） | 大模型预训练 |
| **推理机型** | 高密度部署 | 在线推理服务 |
| **预处理机型** | 大核高 IO | 数据预处理 |

---

## 5. 硬件故障域设计

### 5.1 分层容灾

```
单机故障 → 机架故障 → 机柜故障 → 机房故障
```

每一层独立容灾，防止级联故障。

### 5.2 冗余设计

- 电源冗余（双路供电）
- 风扇散热冗余
- 硬件 BMC 管理（IPMI）

### 5.3 硬件生命周期管理

- IPMI 远程管理
- 硬件健康巡检
- 硬盘/GPU 坏件预测（SMART/dcgmi）

---

## 6. Linux 高性能内核

### 6.1 进程与调度子系统

#### 6.1.1 调度器

- **CFS 调度器**：完全公平调度、虚拟运行时间
- **实时调度**：SCHED_FIFO/SCHED_RR、AI 任务调度优先级
- **任务绑核**：CPU core 绑定、GPU 亲和性、中断绑核隔离

#### 6.1.2 进程管理

- 进程/线程/轻量级进程
- 用户态/内核态切换、上下文切换开销
- AI 任务独占核心策略

#### 6.1.3 Cgroup v1/v2

- **全量子系统**：cpu、cpuset、memory、blkio、devices、hugetlb
- **资源限制**：软限制/硬限制、资源抢占、OOM 控制
- **容器隔离**：资源边界、逃逸风险

### 6.2 内存子系统

#### 6.2.1 内存管理

- 虚拟内存、物理内存、页表、内存碎片
- Slab 缓存机制

#### 6.2.2 OOM 机制

- OOM killer 触发逻辑、权重调节
- AI 大内存进程保护策略：`oom_score_adj`

#### 6.2.3 Swap 与调优

**AI 集群必须禁用 Swap**：

```bash
swapoff -a
sed -i 's/^UUID.*swap/#&/' /etc/fstab
```

关键参数：
| 参数 | 值 | 说明 |
|------|-----|------|
| `vm.swappiness` | 0 | 禁用交换倾向 |
| `vm.zone_reclaim_mode` | 1 | NUMA 本地回收 |
| `vm.min_free_kbytes` | 262144 | 最小空闲内存 |
| `vm.nr_hugepages` | 524288 | 大页内存预留 |

#### 6.2.4 脏页管理

- 脏页回刷机制
- PageCache 管控
- 内存溢出防护

### 6.3 网络子系统

#### 6.3.1 TCP/IP 协议栈

- 内核网络协议栈、四层模型
- **拥塞控制算法**：CUBIC/BBR 选型、AI 集群 BBR 调优

#### 6.3.2 网卡中断优化

- **RSS**：接收端缩放
- **RPS**：接收数据包 Steering
- **XPS**：发送数据包 Steering
- 中断队列隔离

#### 6.3.3 内核网络关键参数

```bash
# 连接数优化
net.core.somaxconn = 65535
net.ipv4.tcp_tw_reuse = 1

# 缓冲区优化
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
```

#### 6.3.4 虚拟网络

- bridge、macvlan、ipvlan
- overlay 网络底层实现（VXLAN）

### 6.4 IO 子系统

#### 6.4.1 块设备调度

- **调度器选型**：mq-deadline/none/kyber
- 场景适配：NVMe 高性能选 none

#### 6.4.2 文件系统缓存

- PageCache、buffer cache
- 脏页刷盘策略

#### 6.4.3 IO 限流与隔离

- 磁盘队列深度
- IOPS/带宽隔离设计
- NVMe 高性能读写、RAID 冗余

### 6.5 内核观测与排障工具

#### 6.5.1 性能剖析

- **perf**：CPU 性能分析
- **ebpf**：动态内核追踪
- **ftrace**：内核函数追踪
- **strace**：系统调用追踪

#### 6.5.2 硬件排查

```bash
nvidia-smi topo -m     # GPU 拓扑
dcgmi health status    # GPU 健康状态
ibstat                 # InfiniBand 状态
rdma link show         # RDMA 链路
numastat               # NUMA 统计
lscpu -e               # CPU 拓扑
```

#### 6.5.3 日志与故障

- `dmesg`：内核日志
- `ss`：socket 统计
- `lsof`：打开文件
- GPU 掉卡、NCCL 通信报错定位方法论

---

> 🔑 **核心要点**：Linux 内核是 AI Infra 架构师的底层底牌。CFS 调度器、Cgroup 隔离、Swap 禁用、HugePage 配置、网络调优、内核观测工具是日常运维和故障排查的必备技能。
