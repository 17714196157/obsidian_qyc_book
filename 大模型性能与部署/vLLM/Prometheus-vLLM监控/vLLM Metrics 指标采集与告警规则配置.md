---
title: "vLLM Metrics 指标采集与告警规则配置"
source: "https://mp.weixin.qq.com/s/AYEhI7ZDGsHZlybozWbFKw"
author:
  - "[[平凡小代]]"
published:
created: 2026-07-09
description:
tags:
  - clippings
  - vllm
  - prometheus
  - monitoring
  - metrics
  - kubernetes
  - 告警
---

## 前言

在前面的 GPU 监控中，我们通过 DCGM Exporter 获取了 GPU 利用率、显存、温度和功耗等硬件层指标。但是，只监控 GPU 还无法完整判断一个大模型推理服务是否健康。

例如，当用户感觉模型响应变慢时，GPU 利用率可能仍然正常。真正的问题可能是：

- 请求已经在 vLLM 中排队；
- KV Cache 使用率过高；
- 输入 Prompt 太长，导致 Prefill 阶段耗时增加；
- 首个 Token 返回时间过长；
- Decode 阶段生成 Token 的速度变慢；
- vLLM 发生了 Preemption；
- 请求由于内部错误而结束。

因此，除了 DCGM Exporter 的 GPU 指标，还需要采集 vLLM 自身暴露的运行指标。

本文主要完成以下内容：

1. 验证 vLLM 是否正常暴露 `/metrics`。
2. 编写 ServiceMonitor，让 Prometheus 采集 vLLM 指标。
3. 解决 Prometheus 跨命名空间发现目标时的 RBAC 权限问题。
4. 介绍 vLLM 请求处理流程以及常见指标。
5. 使用 PromQL 计算请求排队时间、TTFT、ITL 和端到端延迟。
6. 编写 vLLM 对应的 PrometheusRule 告警规则。
7. 验证 PrometheusRule 是否成功加载。

---

## 一、vLLM Metrics 与请求处理流程

### 1.1 vLLM 不需要单独部署 Exporter

DCGM 需要通过 DCGM Exporter 将 GPU 数据转换成 Prometheus 指标，但 vLLM 不需要额外部署一个独立的"vllm-exporter"。vLLM 的 OpenAI Compatible API Server 本身就会通过 `/metrics` 接口暴露 Prometheus 格式的指标。

默认情况下，vLLM 服务监听 `8000` 端口，因此指标地址通常为：

```
http://<vllm-service>:8000/metrics
```

所以本文采集的是：vLLM API Server 自己暴露的 Metrics，而不是一个单独部署的 vLLM Exporter。

### 1.2 vLLM 如何处理一次请求

在理解具体指标之前，需要先了解 vLLM 处理请求的大致流程：

```
客户端发送请求
    ↓
请求进入 vLLM
    ↓
请求进入等待队列
    ↓
Scheduler 选择请求并组成 Batch
    ↓
Prefill：处理输入 Prompt
    ↓
Decode：逐个生成输出 Token
    ↓
请求结束并返回结果
```

vLLM 官方将 Metrics 大致分成两类：

- **Server-level Metrics：** 表示引擎整体状态，通常使用 Gauge 或 Counter，例如正在运行的请求数、等待请求数和 KV Cache 使用率。
- **Request-level Metrics：** 表示单个请求的长度和延迟分布，通常使用 Histogram，例如排队时间、TTFT、ITL 和端到端延迟。

Server-level Metrics 通常用来解释 Request-level Metrics 为什么发生变化。

### 1.3 Prefill、Decode 和 KV Cache

#### Prefill 阶段

Prefill 可以理解为"模型阅读用户输入"的过程。例如，用户输入了一个很长的 Prompt，vLLM 需要先处理这些输入 Token，并生成后续 Decode 所需的 KV Cache。

Prompt 越长，Prefill 阶段通常越重，可以重点关注：

```
vllm:request_prompt_tokens
vllm:request_prefill_time_seconds
vllm:time_to_first_token_seconds
```

#### Decode 阶段

Decode 可以理解为"模型逐个生成输出 Token"的过程。大模型通常不是一次性生成整段回答，而是一个 Token 一个 Token 地生成，所以 Decode 性能会直接影响流式输出速度。可以关注：

```
vllm:request_decode_time_seconds
vllm:inter_token_latency_seconds
vllm:request_time_per_output_token_seconds
vllm:generation_tokens_total
```

#### KV Cache

KV Cache 用来保存已经计算过的 Attention Key 和 Value，避免模型在生成每个新 Token 时重新计算全部历史上下文。KV Cache 会占用 GPU 显存。随着并发数、Prompt 长度和输出长度增加，KV Cache 压力也会增加。可以重点关注：

```
vllm:kv_cache_usage_perc
vllm:num_preemptions_total
vllm:num_requests_waiting
```

其中：

```
vllm:kv_cache_usage_perc = 1
```

表示 KV Cache 已使用 100%。

---

## 二、采集 vLLM 的 Metrics 指标

### 2.1 查看 vLLM Service

查看当前 vLLM 服务的 service。执行：

```bash
kubectl -n ai-demo get svc qwen-demo -o yaml
```

当前 Service 的关键内容如下：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: qwen-demo
  namespace: ai-demo
  labels:
    aiinfra.example.com/model: qwen2.5
    aiinfra.example.com/runtime: vllm
    aiinfra.example.com/scheduler: volcano
    aiinfra.example.com/team: infra
    app.kubernetes.io/instance: qwen-demo
    app.kubernetes.io/managed-by: vllmservice-operator
    app.kubernetes.io/name: vllmservice
spec:
  ports:
    - name: http
      port: 8000
      protocol: TCP
      targetPort: http
  selector:
    app.kubernetes.io/instance: qwen-demo
    app.kubernetes.io/name: vllmservice
  type: ClusterIP
```

编写 ServiceMonitor 时，需要重点关注以下内容：

| Service 字段 | 当前值 | ServiceMonitor 对应字段 |
| --- | --- | --- |
| Service 名称 | `qwen-demo` | 用于识别目标和编写查询 |
| Service 命名空间 | `ai-demo` | `namespaceSelector.matchNames` |
| Service 标签 | `app.kubernetes.io/instance=qwen-demo` | `selector.matchLabels` |
| Service 端口名称 | `http` | `endpoints.port` |
| Service 端口 | `8000` | 由端口名称间接匹配 |
| Metrics 路径 | `/metrics` | `endpoints.path` |

### 2.2 直接访问 /metrics

将 Service 的 `8000` 端口转发到本机：

```bash
kubectl -n ai-demo port-forward service/qwen-demo 8000:8000
```

保持当前终端运行，然后新开一个终端，查看 vLLM 指标：

```bash
curl http://127.0.0.1:8000/metrics
```

正常情况下，可以看到类似内容：

```
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="qwen2.5-1.5b-instruct"} 0

# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="qwen2.5-1.5b-instruct"} 0

# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="qwen2.5-1.5b-instruct"} 0
```

如果 `/metrics` 能够正常返回结果，说明：

```
vLLM Pod
    ↓
Service/qwen-demo
    ↓
8000/metrics
```

这部分链路已经正常。

### 2.3 检查 Prometheus 的 ServiceMonitor 选择范围

执行：

```bash
kubectl -n monitoring get prometheus k8s -o yaml \
  | grep -A20 -E "serviceMonitorSelector|serviceMonitorNamespaceSelector"
```

当前环境的配置为：

```yaml
serviceMonitorNamespaceSelector: {}
serviceMonitorSelector: {}
```

含义如下：

| 字段 | 当前配置含义 |
| --- | --- |
| `serviceMonitorNamespaceSelector: {}` | Prometheus 可以发现所有命名空间里的 ServiceMonitor |
| `serviceMonitorSelector: {}` | Prometheus 选择发现范围内的所有 ServiceMonitor |

因此，可以直接把 vLLM 的 ServiceMonitor 创建在 `ai-demo` 命名空间中。

Prometheus Operator 中的 ServiceMonitor 负责描述监控目标，Prometheus CR 中的选择器负责决定哪些 ServiceMonitor 会被当前 Prometheus 使用。

### 2.4 编写 ServiceMonitor

创建文件：

```bash
vim qwen-demo-servicemonitor.yaml
```

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: qwen-demo
  namespace: ai-demo
  labels:
    app: qwen-demo
spec:
  selector:
    matchLabels:
      app.kubernetes.io/instance: qwen-demo
      app.kubernetes.io/name: vllmservice
  namespaceSelector:
    matchNames:
      - ai-demo
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
      scrapeTimeout: 10s
```

配置中的匹配关系如下：

```
ServiceMonitor.spec.selector.matchLabels
    ↓
匹配 Service.metadata.labels
    ↓
找到 Service/qwen-demo
    ↓
ServiceMonitor.endpoints.port=http
    ↓
匹配 Service.spec.ports[].name=http
    ↓
访问 Service 后端 Endpoint 的 8000/metrics
```

`metadata.labels.app=qwen-demo` 是 ServiceMonitor 自己的标签。当前 Prometheus 配置为：

```yaml
serviceMonitorSelector: {}
```

所以不会限制 ServiceMonitor 标签。如果其他环境中的 Prometheus 配置为：

```yaml
serviceMonitorSelector:
  matchLabels:
    release: prometheus
```

那么 ServiceMonitor 还需要添加：

```yaml
metadata:
  labels:
    release: prometheus
```

应用上面编写的 yaml 文件，查看 ServiceMonitor：

```bash
kubectl -n ai-demo get servicemonitor qwen-demo
```

### 2.5 解决跨命名空间 RBAC 权限问题

创建 ServiceMonitor 后，Prometheus 日志中出现了下面的错误：

```
pods is forbidden:
User "system:serviceaccount:monitoring:prometheus-k8s"
cannot list resource "pods" in the namespace "ai-demo"
```

同时还出现：

```
services is forbidden
endpoints is forbidden
```

这说明：

```
Prometheus 已经尝试发现 ai-demo 中的目标
```

但是：

```
Prometheus 使用的 ServiceAccount 没有权限读取 ai-demo 中的
Pod、Service、Endpoint 或 EndpointSlice
```

这里需要区分两个 ServiceAccount：

- Prometheus Operator 的 ServiceAccount：负责管理 Prometheus 等 CR。
- Prometheus Pod 使用的 ServiceAccount：负责通过 Kubernetes API 发现抓取目标。

当前报错中的用户为：

```
system:serviceaccount:monitoring:prometheus-k8s
```

因此，需要给 `monitoring/prometheus-k8s` 授予读取 `ai-demo` 目标资源的权限。

Prometheus Operator 官方说明，Prometheus Pod 本身需要 Kubernetes API 权限才能完成目标发现；较新的 Kubernetes 和 Prometheus Operator 环境还需要读取 EndpointSlice。

创建 RBAC 的 yaml 文件：

```bash
vim prometheus-ai-demo-rbac.yaml
```

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: prometheus-scrape-discovery
  namespace: ai-demo
rules:
  - apiGroups:
      - ""
    resources:
      - pods
      - services
      - endpoints
    verbs:
      - get
      - list
      - watch

  - apiGroups:
      - discovery.k8s.io
    resources:
      - endpointslices
    verbs:
      - get
      - list
      - watch
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prometheus-scrape-discovery
  namespace: ai-demo
subjects:
  - kind: ServiceAccount
    name: prometheus-k8s
    namespace: monitoring
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: prometheus-scrape-discovery
```

应用配置：

```bash
kubectl apply -f prometheus-ai-demo-rbac.yaml
```

验证 Prometheus ServiceAccount 权限：

```bash
kubectl auth can-i list pods \
  -n ai-demo \
  --as=system:serviceaccount:monitoring:prometheus-k8s
```
```bash
kubectl auth can-i list services \
  -n ai-demo \
  --as=system:serviceaccount:monitoring:prometheus-k8s
```
```bash
kubectl auth can-i list endpoints \
  -n ai-demo \
  --as=system:serviceaccount:monitoring:prometheus-k8s
```
```bash
kubectl auth can-i list endpointslices.discovery.k8s.io \
  -n ai-demo \
  --as=system:serviceaccount:monitoring:prometheus-k8s
```

正常情况下都应该返回：

```
yes
```

### 2.6 在 Prometheus 中验证

打开 prometheus 的 web 页面，查看对应的 Targets
![[file-20260709175644586.png]]



然后进入 Prometheus 查询页面，执行：

```promql
up{namespace="ai-demo"}
```

继续查询：

```promql
vllm:num_requests_running
```
```promql
vllm:num_requests_waiting
```
```promql
vllm:kv_cache_usage_perc
```

如果能查到结果，说明下面这条链路已经打通：

```
vLLM /metrics
    ↓
Service
    ↓
ServiceMonitor
    ↓
Kubernetes 服务发现
    ↓
Prometheus
```

---

## 三、vLLM 常见 Metrics 指标说明

### 3.1 Gauge、Counter 和 Histogram

| 类型 | 特点 | vLLM 示例 |
| --- | --- | --- |
| Gauge | 表示当前状态，可以增加也可以减少 | 当前运行请求数、等待请求数、KV Cache 使用率 |
| Counter | 累计值，通常只增加，进程重启后可能归零 | Token 总数、请求完成数、Preemption 次数 |
| Histogram | 记录大量事件的分布 | 排队时间、TTFT、ITL、请求延迟 |

Counter 通常不能直接使用当前累计值判断吞吐，应配合：

```promql
rate()
```

或者：

```promql
increase()
```

Histogram 通常会展开为三类时间序列：

```
<指标名>_bucket
<指标名>_sum
<指标名>_count
```

例如：

```
vllm:request_queue_time_seconds_bucket
vllm:request_queue_time_seconds_sum
vllm:request_queue_time_seconds_count
```

### 3.2 调度和引擎状态指标

vLLM 当前版本中的引擎和请求指标通常带有：

```
model_name
engine
```

标签。官方源码将这两个标签作为每个引擎指标的基础标签。

| 指标 | 类型 | 含义 | 重点用途 |
| --- | --- | --- | --- |
| `vllm:engine_sleep_state` | Gauge | vLLM Engine 睡眠状态 | 判断 Engine 是否处于 Awake |
| `vllm:num_requests_running` | Gauge | 当前正在执行 Batch 中的请求数 | 观察实时并发 |
| `vllm:num_requests_waiting` | Gauge | 当前等待调度的请求数 | 判断是否出现积压 |
| `vllm:num_requests_waiting_by_reason` | Gauge | 按原因拆分的等待请求数 | 区分容量不足或临时约束 |
| `vllm:kv_cache_usage_perc` | Gauge | KV Cache 已用比例，1 表示 100% | 判断 KV Cache 压力 |

#### engine_sleep_state

该指标通常包含：

```
sleep_state="awake"
sleep_state="weights_offloaded"
sleep_state="discard_all"
```

其中：

```
sleep_state="awake" 且值为 1
```

表示 Engine 处于正常唤醒状态。如果主动使用了 vLLM Sleep Mode，那么 Engine 进入睡眠状态可能是预期行为，不能直接当成异常。

#### num_requests_running

表示已经被 Scheduler 选中、当前正在模型执行 Batch 中处理的请求数。

值较高并不一定异常，它通常说明服务正在处理并发请求。

#### num_requests_waiting

表示已经进入 vLLM，但还没有获得调度执行机会的请求数。

短时间出现等待请求不一定代表故障；如果等待数量持续增加，则通常说明：

- 请求到达速度超过处理速度；
- 当前实例容量不足；
- KV Cache 压力过高；
- Prompt 或输出过长；
- GPU 性能不足；
- 最大并发和调度参数需要调整。

vLLM 官方的 KEDA 扩缩容示例同样使用 `vllm:num_requests_waiting` 作为扩容依据。

#### kv_cache_usage_perc

该指标的取值范围通常为：

```
0～1
```

例如：

```
0.8 = 80%
0.9 = 90%
1.0 = 100%
```

KV Cache 高使用率本身不一定立即导致故障，但如果同时出现：

```
num_requests_waiting 持续增加
num_preemptions_total 增长
TTFT 变高
```

就很可能意味着实例容量不足。

### 3.3 Token 和请求吞吐指标

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `vllm:prompt_tokens_total` | Counter | 已处理的输入 Token 累计数量 |
| `vllm:generation_tokens_total` | Counter | 已生成的输出 Token 累计数量 |
| `vllm:request_success_total` | Counter | 已结束请求数量，按 `finished_reason` 分类 |
| `vllm:num_preemptions_total` | Counter | Engine 发生 Preemption 的累计次数 |
| `vllm:prompt_tokens_cached_total` | Counter | 被缓存命中的 Prompt Token 数量 |

#### prompt_tokens_total

表示 vLLM 累计处理了多少输入 Token。

它本身是累计值，通常使用下面的 PromQL 计算输入 Token 吞吐：

```promql
rate(vllm:prompt_tokens_total[5m])
```

表示最近 5 分钟平均每秒处理多少输入 Token。

#### generation_tokens_total

表示 vLLM 累计生成了多少输出 Token。输出 Token 吞吐可以计算为：

```promql
rate(vllm:generation_tokens_total[5m])
```

#### request_success_total

虽然指标名称中包含 `success`，但当前 vLLM 会通过 `finished_reason` 记录请求结束原因。

较新的 vLLM 版本可能包含：

```
stop
length
abort
error
repetition
```

其中：

- `stop`：正常遇到停止条件；
- `length`：达到最大 Token 数或上下文长度限制；
- `abort`：请求被客户端或系统中止；
- `error`：请求级内部错误；
- `repetition`：检测到重复 Token 模式。

实际值必须以 `/metrics` 中的 `finished_reason` 标签为准。

#### num_preemptions_total

当前 KV Cache 或调度资源不足时，vLLM 暂停某些请求，并在后续重新计算或恢复这些请求。偶尔出现一次不一定代表服务故障，但如果持续增加，通常说明：

- KV Cache 空间不足；
- 并发数过高；
- Prompt 或输出过长；
- `max_num_seqs` 设置过大；
- `max_num_batched_tokens` 需要调整；
- 需要增加模型副本或使用更多 GPU。

vLLM 官方优化文档也建议通过 Prometheus 指标观察 Preemption，并根据情况调整 KV Cache 和调度参数。

### 3.4 请求延迟指标

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `vllm:request_queue_time_seconds` | Histogram | 请求在等待队列中停留的时间 |
| `vllm:time_to_first_token_seconds` | Histogram | 从请求到达至首个 Token 返回的时间，即 TTFT |
| `vllm:inter_token_latency_seconds` | Histogram | 相邻输出 Token 之间的时间间隔，即 ITL |
| `vllm:request_time_per_output_token_seconds` | Histogram | 每个请求平均生成一个输出 Token 的耗时，即请求级 TPOT |
| `vllm:e2e_request_latency_seconds` | Histogram | 请求从到达至完成的总耗时 |
| `vllm:request_prefill_time_seconds` | Histogram | 请求在 Prefill 阶段的耗时 |
| `vllm:request_decode_time_seconds` | Histogram | 请求在 Decode 阶段的耗时 |
| `vllm:request_inference_time_seconds` | Histogram | 请求处于 Running 阶段的耗时 |

vLLM 官方当前 Production Metrics 中提供了这些请求级延迟指标。

#### Queue Time

Queue Time 表示请求进入 vLLM 后，真正开始执行之前等待了多久。

如果 Queue Time 持续升高，通常说明服务吞吐能力已经低于请求到达速度。

#### TTFT

TTFT 的全称是 Time To First Token。它表示从请求进入 vLLM 前端并开始处理，到 vLLM 前端获得第一个输出 Token 所经历的时间。该指标主要反映 vLLM 服务内部的排队和首 Token 生成延迟，通常不包含客户端网络、外部网关和上游应用产生的全部耗时。

TTFT 会受到以下因素影响：

- 排队时间；
- Prompt 长度；
- Prefill 速度；
- GPU 负载；
- KV Cache 状态；
- 并发请求数量。

#### ITL

ITL 的全称是 Inter-Token Latency。它表示流式输出过程中，相邻两个 Token 之间的时间间隔。

ITL 越高，用户越容易感觉模型输出断断续续。

#### TPOT

TPOT 的全称是 Time Per Output Token。`request_time_per_output_token_seconds` 通常表示每个请求平均生成一个输出 Token 所需要的时间。

ITL 和 TPOT 都能反映 Decode 性能，但统计口径并不完全相同：

- ITL 更关注每两个相邻 Token 之间的间隔分布；
- 请求级 TPOT 更关注每个请求平均生成一个 Token 的耗时。

#### E2E Latency

E2E Latency 表示从请求进入 vLLM 前端开始处理，到 vLLM 前端完成该请求所经历的总时间。它属于服务端观测指标，不应直接等同于客户端测得的完整请求耗时。它大致受到以下部分影响：

```
排队时间
+
Prefill 时间
+
Decode 时间
+
其他请求处理开销
```

需要注意，E2E Latency 和输出 Token 数量强相关。一个输出 500 个 Token 的请求，通常天然比只输出 20 个 Token 的请求耗时更长。因此，不能只看 E2E 延迟，还需要结合：

```
request_generation_tokens
request_prompt_tokens
TTFT
ITL 或 TPOT
```

一起判断。

### 3.5 Prefix Cache 指标

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `vllm:prefix_cache_queries_total` | Counter | 参与 Prefix Cache 查询的 Token 数 |
| `vllm:prefix_cache_hits_total` | Counter | 从 Prefix Cache 命中的 Token 数 |
| `vllm:prompt_tokens_cached_total` | Counter | 本地和外部缓存命中的 Prompt Token 数 |

需要注意，`queries` 和 `hits` 统计的是 Token 数量，不是 HTTP 请求数量。

例如，同一个系统提示词被大量请求重复使用：

```
你是一个专业的运维助手……
```

第一次请求需要完整计算这个前缀。后续请求如果命中 Prefix Cache，vLLM 可以直接复用已经生成的 KV Cache，减少重复 Prefill 计算。Token 级 Prefix Cache 命中率可以计算为：

```promql
100 *
sum by (namespace, service, model_name, engine) (
  rate(vllm:prefix_cache_hits_total[5m])
)
/
clamp_min(
  sum by (namespace, service, model_name, engine) (
    rate(vllm:prefix_cache_queries_total[5m])
  ),
  0.001
)
```

### 3.6 corrupted_requests 指标

```
vllm:corrupted_requests_total
```

表示检测到 Logits 中包含 NaN 的异常请求累计数量。

这通常不是普通的 HTTP 参数错误，而可能与以下问题有关：

- 模型计算出现数值异常；
- Attention 或 CUDA Kernel 异常；
- 量化配置问题；
- 模型权重或运行时异常；
- GPU 计算产生异常值。

需要特别注意：当前 vLLM 源码中，该指标只有在启用对应的 NaN 检测功能后才会注册，例如启用：

```
VLLM_COMPUTE_NANS_IN_LOGITS
```

因此，编写告警前必须先确认当前 `/metrics` 中是否存在：

```
vllm:corrupted_requests_total
```

如果指标不存在，不要误认为 Prometheus 采集失败。

---

## 四、vLLM 常用 PromQL

### 4.1 查看当前运行和等待请求

当前正在运行的请求：

```promql
vllm:num_requests_running
```

当前正在等待的请求：

```promql
vllm:num_requests_waiting
```

按模型和 Engine 汇总：

```promql
sum by (namespace, service, model_name, engine) (
  vllm:num_requests_waiting
)
```

### 4.2 计算输入和输出 Token 吞吐

输入 Token 每秒吞吐：

```promql
sum by (namespace, service, model_name, engine) (
  rate(vllm:prompt_tokens_total[5m])
)
```

输出 Token 每秒吞吐：

```promql
sum by (namespace, service, model_name, engine) (
  rate(vllm:generation_tokens_total[5m])
)
```

请求完成速率：

```promql
sum by (namespace, service, model_name, engine) (
  rate(vllm:request_success_total[5m])
)
```

Prometheus 官方建议 Counter 使用 `rate()` 计算每秒平均增长率。`rate()` 还会处理目标重启导致的 Counter 重置；聚合 Counter 时，应先执行 `rate()`，再执行 `sum()`。

### 4.3 计算 P95 排队时间

```promql
histogram_quantile(
  0.95,
  sum by (le, namespace, service, model_name, engine) (
    rate(vllm:request_queue_time_seconds_bucket[5m])
  )
)
```

这段 PromQL 可以分成三层理解。

#### 第一层：获取最近 5 分钟 Bucket 增长率

```promql
rate(vllm:request_queue_time_seconds_bucket[5m])
```

表示计算每个 Histogram Bucket 在最近 5 分钟内的平均增长速率。

#### 第二层：聚合 Bucket

```promql
sum by (le, namespace, service, model_name, engine) (...)
```

其中：

- `le` 表示每个 Bucket 的上限；
- `namespace` 和 `service` 保留 Kubernetes 目标信息；
- `model_name` 和 `engine` 保留 vLLM 模型和 Engine 信息。

经典 Histogram 在使用 `histogram_quantile()` 时必须保留 `le` 标签。

#### 第三层：计算 P95

```promql
histogram_quantile(0.95, ...)
```

表示估算第 95 百分位。P95 排队时间为 2 秒，表示：

```
最近统计窗口内，大约 95% 的请求排队时间不超过 2 秒。
```

P95 不是最大值，也不代表每个请求都等待了 2 秒。

### 4.4 计算 TTFT P95

```promql
histogram_quantile(
  0.95,
  sum by (le, namespace, service, model_name, engine) (
    rate(vllm:time_to_first_token_seconds_bucket[5m])
  )
)
```

### 4.5 计算 ITL P95

```promql
histogram_quantile(
  0.95,
  sum by (le, namespace, service, model_name, engine) (
    rate(vllm:inter_token_latency_seconds_bucket[5m])
  )
)
```

### 4.6 计算 E2E Latency P95

```promql
histogram_quantile(
  0.95,
  sum by (le, namespace, service, model_name, engine) (
    rate(vllm:e2e_request_latency_seconds_bucket[5m])
  )
)
```

### 4.7 多副本场景需要保留 Pod 标签

当前演示环境只有一个 vLLM 副本，因此使用：

```promql
sum by (le, namespace, service, model_name, engine)
```

可以得到整个 Service 的延迟。如果后续扩容到多个副本，且需要分别观察每个 Pod，可以改成：

```promql
sum by (
  le,
  namespace,
  service,
  pod,
  model_name,
  engine
) (
  rate(vllm:time_to_first_token_seconds_bucket[5m])
)
```

否则多个副本中相同 `model_name` 和 `engine` 的 Histogram 会被聚合成服务级结果。

---

## 五、编写 vLLM PrometheusRule

### 5.1 告警阈值说明

下面的阈值主要面向当前单卡、小模型测试环境：

```
Qwen2.5-1.5B-Instruct
单个 vLLM 副本
单 GPU
```

这些数值不是 vLLM 官方规定的统一生产阈值。生产环境需要根据以下因素调整：

- 模型大小；
- GPU 型号；
- 最大上下文长度；
- 请求并发；
- 输入和输出 Token 长度；
- 是否使用流式输出；
- 业务 SLO；
- 压测结果；
- 历史监控基线。

在编写规则前，先在 Prometheus 中确认目标标签：

```promql
up{namespace="ai-demo"}
```

重点确认是否存在：

```
service="qwen-demo"
```

如果当前环境没有 `service` 标签，需要根据实际标签修改 Target 告警表达式。

### 5.2 完整 PrometheusRule

创建 Prometheus-rule 文件：

```bash
vim vllm-qwen-alert-rules.yaml
```

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-qwen-alert-rules
  namespace: monitoring
  labels:
    release: prometheus
spec:
  groups:
    - name: vllm-qwen.rules
      rules:
        # Target 存在，但是 Prometheus 最近一次抓取失败。
        - alert: VllmTargetDown
          expr: |
            up{
              namespace="ai-demo",
              service="qwen-demo"
            } == 0
          for: 2m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM Metrics Target 不可用"
            description: "namespace={{ $labels.namespace }} service={{ $labels.service }} 已连续 2 分钟抓取失败。请检查 vLLM Pod、Service、ServiceMonitor、RBAC、8000 端口和 /metrics 接口。"

        # Target 已经完全从 Prometheus 服务发现中消失。
        - alert: VllmTargetMissing
          expr: |
            absent(
              up{
                namespace="ai-demo",
                service="qwen-demo"
              }
            )
          for: 5m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM Metrics Target 消失"
            description: "Prometheus 已连续 5 分钟没有发现 ai-demo/qwen-demo Target。请检查 ServiceMonitor 选择器、Service、Endpoint、EndpointSlice 和 RBAC。"

        # 如果主动使用 vLLM Sleep Mode，应按实际需求禁用或修改此规则。
        - alert: VllmEngineNotAwake
          expr: |
            vllm:engine_sleep_state{
              sleep_state="awake"
            } != 1
          for: 5m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM Engine 不处于 Awake 状态"
            description: "namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 已连续 5 分钟不处于 Awake 状态。若未主动启用 Sleep Mode，请检查 Engine 状态和 vLLM 日志。"

        # Warning：等待请求数为 1～5。
        - alert: VllmRequestWaiting
          expr: |
            (
              vllm:num_requests_waiting > 0
            )
            and
            (
              vllm:num_requests_waiting <= 5
            )
          for: 5m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM 请求出现持续排队"
            description: "namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 等待请求数大于 0 已持续 5 分钟，当前 waiting={{ $value }}。请检查并发、KV Cache、GPU 利用率以及输入输出长度。"

        # Critical：等待请求数超过 5。
        - alert: VllmRequestWaiting
          expr: vllm:num_requests_waiting > 5
          for: 5m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM 请求严重积压"
            description: "namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 等待请求数超过 5 已持续 5 分钟，当前 waiting={{ $value }}。当前实例处理能力可能不足。"

        # Warning：P95 排队时间大于 2 秒且不超过 10 秒。
        - alert: VllmQueueTimeHighP95
          expr: |
            (
              histogram_quantile(
                0.95,
                sum by (
                  le,
                  namespace,
                  service,
                  model_name,
                  engine
                ) (
                  rate(
                    vllm:request_queue_time_seconds_bucket[5m]
                  )
                )
              ) > 2
            )
            and
            (
              histogram_quantile(
                0.95,
                sum by (
                  le,
                  namespace,
                  service,
                  model_name,
                  engine
                ) (
                  rate(
                    vllm:request_queue_time_seconds_bucket[5m]
                  )
                )
              ) <= 10
            )
          for: 5m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM 请求排队时间 P95 较高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟 P95 Queue Time 超过 2 秒，当前值为 {{ $value | printf "%.2f" }} 秒。'

        # Critical：P95 排队时间超过 10 秒。
        - alert: VllmQueueTimeHighP95
          expr: |
            histogram_quantile(
              0.95,
              sum by (
                le,
                namespace,
                service,
                model_name,
                engine
              ) (
                rate(
                  vllm:request_queue_time_seconds_bucket[5m]
                )
              )
            ) > 10
          for: 5m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM 请求排队时间 P95 严重过高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟 P95 Queue Time 超过 10 秒，当前值为 {{ $value | printf "%.2f" }} 秒。请优先检查实例容量和请求并发。'

        # Warning：KV Cache 使用率大于 80% 且不超过 90%。
        - alert: VllmKvCacheUsageHigh
          expr: |
            (
              vllm:kv_cache_usage_perc > 0.80
            )
            and
            (
              vllm:kv_cache_usage_perc <= 0.90
            )
          for: 5m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM KV Cache 使用率较高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} KV Cache 使用率超过 80% 已持续 5 分钟，当前值为 {{ $value | printf "%.2f" }}。'

        # Critical：KV Cache 使用率超过 90%。
        - alert: VllmKvCacheUsageHigh
          expr: vllm:kv_cache_usage_perc > 0.90
          for: 5m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM KV Cache 使用率严重过高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} KV Cache 使用率超过 90% 已持续 5 分钟，当前值为 {{ $value | printf "%.2f" }}。可能导致排队、TTFT 升高或 Preemption。'

        # 最近 10 分钟发生 Preemption。
        - alert: VllmPreemptionIncreasing
          expr: |
            increase(
              vllm:num_preemptions_total[10m]
            ) > 0
          for: 2m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM 发生 Preemption"
            description: "namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 10 分钟发生 Preemption，增长次数为 {{ $value }}。请检查 KV Cache、并发、上下文长度和调度参数。"

        # Warning：TTFT P95 大于 5 秒且不超过 15 秒。
        - alert: VllmTTFTHighP95
          expr: |
            (
              histogram_quantile(
                0.95,
                sum by (
                  le,
                  namespace,
                  service,
                  model_name,
                  engine
                ) (
                  rate(
                    vllm:time_to_first_token_seconds_bucket[5m]
                  )
                )
              ) > 5
            )
            and
            (
              histogram_quantile(
                0.95,
                sum by (
                  le,
                  namespace,
                  service,
                  model_name,
                  engine
                ) (
                  rate(
                    vllm:time_to_first_token_seconds_bucket[5m]
                  )
                )
              ) <= 15
            )
          for: 10m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM TTFT P95 较高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟 P95 TTFT 超过 5 秒，当前值为 {{ $value | printf "%.2f" }} 秒。请检查排队时间、Prefill、Prompt 长度和 GPU 压力。'

        # Critical：TTFT P95 超过 15 秒。
        - alert: VllmTTFTHighP95
          expr: |
            histogram_quantile(
              0.95,
              sum by (
                le,
                namespace,
                service,
                model_name,
                engine
              ) (
                rate(
                  vllm:time_to_first_token_seconds_bucket[5m]
                )
              )
            ) > 15
          for: 10m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM TTFT P95 严重过高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟 P95 TTFT 超过 15 秒，当前值为 {{ $value | printf "%.2f" }} 秒。用户会明显感到首字响应缓慢。'

        # ITL P95 超过 0.5 秒。
        - alert: VllmInterTokenLatencyHighP95
          expr: |
            histogram_quantile(
              0.95,
              sum by (
                le,
                namespace,
                service,
                model_name,
                engine
              ) (
                rate(
                  vllm:inter_token_latency_seconds_bucket[5m]
                )
              )
            ) > 0.5
          for: 10m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM ITL P95 较高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟 P95 ITL 超过 0.5 秒，当前值为 {{ $value | printf "%.2f" }} 秒。流式输出速度可能明显下降。'

        # E2E 延迟需要结合输出 Token 数量判断。
        - alert: VllmE2ELatencyHighP95
          expr: |
            histogram_quantile(
              0.95,
              sum by (
                le,
                namespace,
                service,
                model_name,
                engine
              ) (
                rate(
                  vllm:e2e_request_latency_seconds_bucket[5m]
                )
              )
            ) > 30
          for: 10m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM E2E 请求延迟 P95 较高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟 P95 E2E Latency 超过 30 秒，当前值为 {{ $value | printf "%.2f" }} 秒。请结合 Prompt 和输出 Token 数量判断。'

        # 最近 5 分钟至少出现一次 error finish reason。
        - alert: VllmRequestErrorDetected
          expr: |
            increase(
              vllm:request_success_total{
                finished_reason="error"
              }[5m]
            ) > 0
          for: 2m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM 请求出现内部错误"
            description: "namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟出现 error 类型的结束请求，增长次数为 {{ $value }}。请检查 vLLM 日志、模型状态、请求参数和内存压力。"

        # 最近 5 分钟 error finish reason 占比超过 5%。
        - alert: VllmRequestErrorRatioHigh
          expr: |
            100 *
            sum by (
              namespace,
              service,
              model_name,
              engine
            ) (
              rate(
                vllm:request_success_total{
                  finished_reason="error"
                }[5m]
              )
            )
            /
            clamp_min(
              sum by (
                namespace,
                service,
                model_name,
                engine
              ) (
                rate(
                  vllm:request_success_total[5m]
                )
              ),
              0.001
            )
            > 5
          for: 10m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM 请求错误率较高"
            description: 'namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟 error 类型请求占比超过 5%，当前值为 {{ $value | printf "%.2f" }}%。'

        # 只有 /metrics 中存在 corrupted_requests_total 时才有效。
        - alert: VllmCorruptedRequestDetected
          expr: |
            increase(
              vllm:corrupted_requests_total[5m]
            ) > 0
          for: 1m
          labels:
            severity: critical
            notify_group: ai-infra
          annotations:
            summary: "vLLM 检测到 Corrupted Request"
            description: "namespace={{ $labels.namespace }} model={{ $labels.model_name }} engine={{ $labels.engine }} 最近 5 分钟检测到 Logits 包含 NaN 的请求，增长次数为 {{ $value }}。请重点检查模型权重、量化、CUDA Kernel 和 GPU 运行状态。"

        # 限定到 qwen-demo，避免匹配 Prometheus 抓取的其他进程。
        - alert: VllmProcessOpenFdsHigh
          expr: |
            process_open_fds{
              namespace="ai-demo",
              service="qwen-demo"
            }
            /
            process_max_fds{
              namespace="ai-demo",
              service="qwen-demo"
            }
            > 0.80
          for: 5m
          labels:
            severity: warning
            notify_group: ai-infra
          annotations:
            summary: "vLLM 进程文件描述符使用率较高"
            description: 'namespace={{ $labels.namespace }} pod={{ $labels.pod }} 文件描述符使用率超过 80%，当前值为 {{ $value | printf "%.2f" }}。请检查长连接数量、连接泄漏和进程 fd limit。'
```

### 5.3 为什么 Warning 和 Critical 使用互斥条件

原始规则中，如果写成：

```promql
vllm:num_requests_waiting > 0
```

以及：

```promql
vllm:num_requests_waiting > 5
```

当等待请求数为 `8` 时，两条规则会同时成立：

```
Warning 告警成立
Critical 告警也成立
```

虽然可以通过 Alertmanager Inhibit Rule 压制 Warning，但 Prometheus 页面中仍然会同时出现两条活动告警。因此，本文将 Warning 改为：

```promql
(vllm:num_requests_waiting > 0)
and
(vllm:num_requests_waiting <= 5)
```

Critical 保持为：

```promql
vllm:num_requests_waiting > 5
```

这样两个阈值互斥：

```
1～5：Warning
大于 5：Critical
```

Queue Time、KV Cache 和 TTFT 的 Warning 与 Critical 也采用了相同方式。

---

## 六、验证 PrometheusRule

### 6.1 检查规则选择器

PrometheusRule 创建成功后，不代表一定会被 Prometheus 加载。先检查 Prometheus：

```bash
kubectl -n monitoring get prometheus k8s -o yaml \
  | grep -A20 -E "ruleSelector|ruleNamespaceSelector"
```

如果配置为：

```yaml
ruleSelector:
  matchLabels:
    release: prometheus
```

那么本文的 PrometheusRule 已经包含：

```yaml
metadata:
  labels:
    release: prometheus
```

如果 `ruleNamespaceSelector` 只选择特定命名空间，需要确认其允许读取 `monitoring` 中的 PrometheusRule。

### 6.2 创建并检查规则

应用上面的 YAML 文件，查看资源：

```bash
kubectl -n monitoring get prometheusrule \
  vllm-qwen-alert-rules
```

查看完整 YAML：

```bash
kubectl -n monitoring get prometheusrule \
  vllm-qwen-alert-rules \
  -o yaml
```

### 6.3 在 Prometheus 中检查规则

浏览器访问 prometheus 的 web 页面，可以查看到对应的告警规则。

[[file-20260709175644618.png|Open: file-20260709175324622.png]]
![[file-20260709175644618.png]]
---

## 七、本文总结

本文完成了从 vLLM `/metrics` 到 Prometheus 告警的完整链路：

```
vLLM OpenAI Compatible API Server
    ↓
直接暴露 8000/metrics
    ↓
Service/qwen-demo
    ↓
ServiceMonitor/qwen-demo
    ↓
Prometheus ServiceAccount RBAC
    ↓
Prometheus 采集指标
    ↓
PromQL 计算队列、吞吐和延迟
    ↓
PrometheusRule 判断异常
```

本文需要重点记住以下内容：

1. vLLM API Server 自身直接暴露 `/metrics`，不需要额外部署独立的 vLLM Exporter。
2. ServiceMonitor 的 `selector.matchLabels` 匹配的是 Service 的 `metadata.labels`。
3. ServiceMonitor 的 `endpoints.port` 填写 Service 端口名称 `http`，不是数字 `8000`。
4. vLLM 不同版本的指标可能发生变化，PromQL 必须以当前 `/metrics` 为准。
5. Counter 实际暴露时通常带 `_total`，应使用 `rate()` 或 `increase()`。
6. `num_requests_waiting` 用于观察请求积压。
7. `kv_cache_usage_perc` 的 `1` 表示 KV Cache 使用率 100%。
8. TTFT 表示首 Token 延迟，ITL 表示相邻 Token 之间的延迟，E2E 表示请求总耗时。
9. Histogram P95 需要使用 `histogram_quantile()`、`rate()` 和 `_bucket` 指标计算。
10. Prefix Cache 的 queries 和 hits 统计的是 Token 数量，不是请求数量。
11. `num_preemptions_total` 持续增长通常说明 KV Cache 或调度容量存在压力。
12. `corrupted_requests_total` 只有在对应 NaN 检测功能启用且指标存在时才适合配置告警。
13. Warning 和 Critical 阈值应尽量设置为互斥，避免同一个问题同时触发两种严重级别。
14. 所有阈值都应通过压测和历史基线调整，不能直接作为所有生产环境的统一标准。
