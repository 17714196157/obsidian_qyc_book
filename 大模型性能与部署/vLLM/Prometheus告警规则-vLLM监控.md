---
tags:
  - vLLM监控
---
### 整体架构图
> **vLLM 说"我怎么样了"，DCGM 说"GPU 怎么样了"，Prometheus 负责"记录下来并判断要不要报警"，Grafana 负责"画出来给人看"。**
> 
```
┌─────────────────────────────────────────────────────────────┐
│                         宿主机 (Host)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   vLLM 服务  │  │  Prometheus │  │      Grafana        │ │
│  │  :8000      │  │   :9090     │  │      :3000          │ │
│  │             │  │             │  │                     │ │
│  │  /metrics ──┼──┼─► scrape    │  │  ◄── query metrics │ │
│  │  (被采集)    │  │  (存储+告警) │  │  (可视化展示)       │ │
│  └─────────────┘  └──────┬──────┘  └─────────────────────┘ │
│                          │                                   │
│  ┌───────────────────────┘                                   │
│  │  DCGM Exporter :9400                                       │
│  │  GPU 硬件指标 ───────────────► Prometheus scrape           │
│  └──────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘

vLLM 推理服务
    │
    ▼  暴露 /metrics
┌─────────────┐
│  vLLM 内部  │  ──► vllm:kv_cache_usage_perc
│  业务指标   │  ──► vllm:num_requests_waiting
│             │  ──► vllm:time_to_first_token_seconds_bucket
│             │  ──► vllm:generation_tokens_total
└─────────────┘
    │
    ▼  Prometheus 每 15s 抓取
┌─────────────┐
│  Prometheus │  ◄── 存储时序数据
│  时序数据库 │  ◄── 执行告警规则 (vllm_alerts.yml)
│             │  ◄── 触发告警 (如 TTFT > 5s)
└─────────────┘
    ▲
    │  同时抓取
    ▼
┌─────────────┐
│ DCGM Exporter│  ──► DCGM_FI_DEV_FB_USED (显存已用)
│  GPU 硬件指标│  ──► DCGM_FI_DEV_FB_FREE (显存空闲)
│             │  ──► 温度、功耗、利用率等
└─────────────┘
    │
    ▼  Grafana 查询展示
┌─────────────┐
│   Grafana   │  ──► 仪表盘 (Dashboard)
│  可视化面板 │  ──► 折线图、热力图、告警状态
│             │  ──► 配置数据源: http://localhost:9090
└─────────────┘
```


### 配置文件
| 告警名                  | 触发条件           | 严重程度     | 典型根因                    |
| -------------------- | -------------- | -------- | ----------------------- |
| `VLLMKVCacheHigh`    | KV Cache > 90% | warning  | 请求过多、max\_model\_len 过大 |
| `VLLMQueueBacklog`   | 排队请求 > 10      | critical | 并发过高、推理速度慢              |
| `VLLMTTFTHigh`       | TTFT P99 > 5s  | warning  | prompt 过长、prefill 耗时    |
| `VLLMThroughputDrop` | 吞吐量降 50%       | critical | GPU 散热降频、硬件异常           |
| `GPUMemoryHigh`      | 显存 > 95%       | critical | 内存泄漏、并发过高               |


**prometheus.yml：**
```
# ============================================
# prometheus.yml - vLLM 生产监控配置
# ============================================

global:
  scrape_interval: 15s
  evaluation_interval: 15s

# 告警规则文件引用
rule_files:
  - "vllm_alerts.yml"

# 告警管理器配置（可选，如需告警通知需额外部署 Alertmanager）
# alerting:
#   alertmanagers:
#     - static_configs:
#         - targets: ['localhost:9093']

scrape_configs:
  # -----------------------
  # 1. vLLM 推理服务指标
  # -----------------------
  - job_name: "vllm"
    static_configs:
      - targets: ["localhost:8102"]
    metrics_path: "/metrics"
    scrape_interval: 15s

  # -----------------------
  # 2. DCGM Exporter - GPU 硬件指标
  # 部署命令:
  #  docker run -d --gpus all \
  # --name dcgm-exporter \
  # --restart unless-stopped \
  # -p 9400:9400 \
  # nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04

  # -----------------------
  - job_name: "dcgm-exporter"
    static_configs:
      - targets: ["localhost:9400"]
    scrape_interval: 15s

  # -----------------------
  # 3. Prometheus 自身监控
  # -----------------------
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
        
```

**vllm_alerts.yml :**
```
# ============================================
# vllm_alerts.yml - vLLM 生产告警规则
# ============================================

groups:
  - name: vllm_memory
    rules:
      # 1. KV Cache 使用率过高 → 服务即将触发抢占
      - alert: VLLMKVCacheHigh
        expr: vllm:kv_cache_usage_perc > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "KV Cache 使用率 {{ $value | humanizePercentage }}"
          description: "KV Cache 使用率超过 90%，新请求可能触发抢占(preemption)，导致请求被反复踢出重排队。建议：检查排队数，若同步上涨则考虑扩容或降低 max_model_len。"

      # 2. 请求排队积压 → 用户体感延迟
      - alert: VLLMQueueBacklog
        expr: vllm:num_requests_waiting > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "请求积压 {{ $value }} 个"
          description: "当前有 {{ $value }} 个请求在等待调度。排队 <5 用户无感，10-20 TTFT P50 明显上涨，>50 用户体感'模型不回了'。"

  - name: vllm_latency
    rules:
      # 3. 首 Token 延迟 P99 超标 → 用户体感"卡住了"
      - alert: VLLMTTFTHigh
        expr: histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTFT P99 超过 5 秒"
          description: "首 Token 延迟 P99 达到 {{ $value }}s。可能原因：prompt 过长、prefill 阶段耗时增加。建议检查调用方 prompt 长度限制。"

  - name: vllm_throughput
    rules:
      # 4. Token 吞吐量骤降 50% → 模型或 GPU 异常（如散热降频）
      - alert: VLLMThroughputDrop
        expr: rate(vllm:generation_tokens_total[5m]) < rate(vllm:generation_tokens_total[30m]) * 0.5
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Token 吞吐量骤降"
          description: "近 5 分钟 Token 生成速率较 30 分钟基准下降超过 50%。可能原因：GPU 散热降频、模型异常。注意：GPU 利用率和进程状态可能仍显示正常。"

  - name: gpu_hardware
    rules:
      # 5. GPU 显存超过 95% → 最后一根稻草
      - alert: GPUMemoryHigh
        expr: DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) > 0.95
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "GPU 显存 >95%"
          description: "GPU 显存使用率超过 95%，即将触发 OOM。建议立即检查是否有内存泄漏或考虑降低并发。"
```

### 部署启动
```bash
# 1. 创建工作目录
mkdir -p /root/vllm-monitoring && cd /root/vllm-monitoring

# 2. 保存上面两个 YAML 文件
# prometheus.yml
# vllm_alerts.yml

# 3. 启动 DCGM Exporter（如需 GPU 硬件监控）
docker run -d \
  --name dcgm-exporter \
  --cap-add SYS_ADMIN \
  --cap-add SYS_PTRACE \
  --security-opt apparmor:unconfined \
  --gpus all \
  --pid host \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04

| 参数                    | 作用                         |
| --------------------- | -------------------------- |
| `--privileged`        | 授予容器完全特权（解决 `non-root` 问题） |
| `--cap-add SYS_ADMIN` | 暴露性能分析指标（日志中的 Warning #2）  |
| `--pid host`          | 共享宿主机 PID 命名空间，DCGM 需要     |
# 查看所有 GPU 相关指标
curl -s http://localhost:9400/metrics | grep -i "gpu\|memory\|fb"


# 4. 启动 Prometheus（修正了原文章的挂载路径错误）
docker run -d --network host \
  --name prometheus \
  -v /root/vllm-monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v /root/vllm-monitoring/vllm_alerts.yml:/etc/prometheus/vllm_alerts.yml \
  prom/prometheus:latest

# 5. 启动 Grafana
docker run -d --network host \
  --name grafana \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana:latest
```

### 状态检查
```bash
# 1. 确认 vLLM /metrics 可访问
curl http://localhost:8000/metrics | head

# 2. 确认 Prometheus targets 全部 UP
open http://localhost:9090/targets

# 3. 确认告警规则加载成功
curl -s http://localhost:9090/api/v1/rules | grep -c '"health":"ok"'
# 输出应 ≥ 5

# 4. 确认 DCGM GPU 指标正常（如已部署）
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_FB_USED

# 5. 访问 Grafana
# http://localhost:3000  账号: admin / admin
```


### 使用演示
```bash
# 打开 Prometheus UI（浏览器）
open http://localhost:9090
```

| 页面路径                            | 作用       | 怎么用                        |
| ------------------------------- | -------- | -------------------------- |
| `http://localhost:9090/targets` | 查看所有监控目标 | 确认 vLLM 和 DCGM 状态都是 **UP** |
| `http://localhost:9090/graph`   | 查询指标、画图表 | 输入 PromQL 表达式查数据           |
| `http://localhost:9090/rules`   | 查看告警规则   | 确认 5 条规则已加载                |
| `http://localhost:9090/alerts`  | 查看当前告警   | 看有没有触发中的告警                 |

![[graph.png]]