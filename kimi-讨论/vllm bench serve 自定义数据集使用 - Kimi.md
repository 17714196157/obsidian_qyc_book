---
title: "自定义数据集使用 - Kimi"
source: "https://www.kimi.com/chat/19d014eb-e052-8cf4-8000-09fa36d3f65f?chat_enter_method=new_chat"
author:
published:
tags:
  - "clippings"
---
 vllm bench serve 如何使用
我来搜索关于 vLLM bench serve 的最新使用方法。
根据搜索结果，我来为你详细介绍 `vllm bench serve` 的使用方法。

## vLLM Bench Serve 使用指南

`vllm bench serve` 是 vLLM 提供的在线服务吞吐量基准测试工具，用于测试已运行的 vLLM 服务器性能。

### 1\. 安装依赖

首先需要安装 benchmark 额外依赖：

```bash
pip install vllm[bench]
```

### 2\. 基本使用流程

**第一步：启动 vLLM 服务**

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

**第二步：运行基准测试**

```bash
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host localhost \
    --port 8000 \
    --random-input-len 32 \
    --random-output-len 4 \
    --num-prompts 5
```

### 3\. 常用参数说明

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--model` | 模型名称（需与 serve 端一致） | `meta-llama/Llama-3.2-1B-Instruct` |
| `--host` | 服务器主机地址 | `localhost` 或 `0.0.0.0` |
| `--port` | 服务器端口 | `8000` |
| `--random-input-len` | 随机输入长度 | `32` |
| `--random-output-len` | 随机输出长度 | `4` |
| `--num-prompts` | 测试请求总数 | `100` |
| `--max-concurrency` | 最大并发请求数 | `32` |
| `--dataset-name` | 数据集类型（random/sharegpt） | `random` |
| `--request-rate` | 请求速率（每秒请求数） | `inf` （无限） |
| `--seed` | 随机种子 | `12345` |

### 4\. 高级用法示例

**使用随机数据集进行高并发测试：**

```bash
vllm bench serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host localhost \
    --port 8000 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 1000 \
    --max-concurrency 32 \
    --seed 12345
```

### 5\. 输出指标说明

运行后会输出以下关键性能指标：

- **Request throughput (req/s)**: 请求吞吐量
- **Input token throughput (tok/s)**: 输入 token 吞吐量
- **Output token throughput (tok/s)**: 输出 token 吞吐量
- **Mean TTFT (ms)**: 首 token 延迟平均值（Time To First Token）
- **Mean TPOT (ms)**: 每输出 token 延迟平均值（Time Per Output Token）
- **Mean ITL (ms)**: token 间延迟平均值（Inter-token Latency）

### 6\. 最佳实践建议

1. **预热测试** ：正式压测前先运行少量请求预热模型
2. **逐步增压** ：从低并发开始，逐步增加并发数观察性能变化
3. **多次测试** ：每个配置运行多次取平均值，减少随机误差
4. **监控资源** ：压测时监控 GPU/CPU/内存使用情况
5. **网络延迟** ：确保客户端到服务器的网络延迟最小化

### 7\. 注意事项

- 确保 vLLM 服务器在指定的 host 和 port 上运行并可访问
- 压测脚本只发送 HTTP 请求，不会加载模型，因此不需要额外 GPU 内存
- 如果使用 Docker 部署，可以在容器内直接运行 benchmark
- 旧版的 `benchmarks/benchmark_serving.py` 脚本已弃用，推荐使用 `vllm bench serve` 命令

更多详细参数可以参考官方文档： `vllm bench serve --help`

\--dataset-name 如何使用自己制作的数据集

我来搜索关于 vLLM bench serve 使用自定义数据集的具体方法。
根据搜索结果，我来详细介绍如何使用自己制作的自定义数据集进行 `vllm bench serve` 测试。

## 使用自定义数据集的方法

### 1\. 数据集格式要求

自定义数据集需要保存为 **JSONL 格式** （每行一个 JSON 对象），必须包含 `prompt` 字段 ：

**基本格式示例** (`data.jsonl`)：

```markdown
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```

### 2\. 基础使用命令

```bash
# 启动 vLLM 服务
vllm serve meta-llama/Llama-3.1-8B-Instruct

# 运行自定义数据集基准测试
vllm bench serve \
    --dataset-name custom \
    --dataset-path /path/to/your/data.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --custom-output-len 256 \
    --num-prompts 100 \
    --max-concurrency 32
```

### 3\. 完整参数说明

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--dataset-name custom` | **必须** ，指定使用自定义数据集 | `custom` |
| `--dataset-path` | **必须** ，自定义数据集文件路径 | `/path/to/data.jsonl` |
| `--custom-output-len` | 每个请求的输出 token 数（默认 256） | `128` |
| `--custom-skip-chat-template` | 跳过应用 chat template（如果数据已包含模板） | 无需值 |
| `--model` | 模型名称（需与 serve 端一致） | `meta-llama/Llama-3.1-8B-Instruct` |
| `--backend` | 后端类型 | `vllm` 或 `openai` |
| `--endpoint` | API 端点 | `/v1/completions` |
| `--num-prompts` | 测试请求数（超过数据条数会循环采样） | `1000` |
| `--max-concurrency` | 最大并发数 | `32` |

### 4\. 高级使用示例

**跳过 Chat Template（数据已格式化）** ：

```bash
vllm bench serve \
    --port 9001 \
    --save-result \
    --save-detailed \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint /v1/completions \
    --dataset-name custom \
    --dataset-path /path/to/your/data.jsonl \
    --custom-skip-chat-template \
    --custom-output-len 128 \
    --num-prompts 80 \
    --max-concurrency 1 \
    --temperature=0.3 \
    --top-p=0.75 \
    --result-dir "./log/"
```

**使用 OpenAI Chat 后端（适用于对话模型）** ：

```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:8003 \
    --model meta-llama/Llama-3.1-8B \
    --dataset-name custom \
    --dataset-path vllm_custom_dataset.jsonl \
    --custom-skip-chat-template \
    --num-prompts 50 \
    --max-concurrency 3
```

### 5\. 多模态自定义数据集（CustomMM）

如果需要测试多模态模型（Vision-Language Models），使用 `custom_mm` 数据集类型 ：

**数据格式** (`mm_data.jsonl`)：

```markdown
{"prompt": "How many animals are present in the given image?", "image_files": ["/path/to/image/folder/horsepony.jpg"]}
{"prompt": "What colour is the bird shown in the image?", "image_files": ["/path/to/image/folder/flycatcher.jpeg"]}
```

**运行命令** ：

```bash
# 启动多模态模型服务
vllm serve Qwen/Qwen2-VL-7B-Instruct

# 运行多模态基准测试
vllm bench serve \
    --save-result \
    --save-detailed \
    --backend openai-chat \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --endpoint /v1/chat/completions \
    --dataset-name custom_mm \
    --dataset-path /path/to/mm_data.jsonl \
    --allowed-local-media-path /path/to/image/folder
```

### 6\. 制作自定义数据集的建议

**简单文本数据集** ：

```python
import json

# 创建简单的问答数据集
data = [
    {"prompt": "解释量子计算的基本原理"},
    {"prompt": "写一个Python函数计算斐波那契数列"},
    {"prompt": "总结人工智能的发展历程"},
    # ... 更多数据
]

# 保存为 JSONL
with open('my_dataset.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

**带预期输出长度的数据集** （用于更精确的控制）：

```markdown
{"prompt": "解释量子计算", "expected_output_len": 100}
{"prompt": "写Python函数", "expected_output_len": 50}
```

**注意事项** ：

- 如果 `--num-prompts` 大于数据条数，vLLM 会循环采样（oversample），除非使用 `--no-oversample`
- 使用 `--save-result` 和 `--save-detailed` 可以保存详细结果供后续分析
- 自定义数据集不支持自动计算 prompt 长度，需要通过 `--custom-output-len` 指定输出长度

并行测试的时候， 我最好监控那些资源情况， 如何方便的监控

编辑

复制

分享

我来搜索关于 vLLM 性能测试时的资源监控方法和工具。

根据搜索结果，我来为你详细介绍 vLLM 并行测试时的资源监控方案，包括关键监控指标和方便的监控工具。

## 一、必须监控的核心资源指标

### 1\. GPU 资源指标（最重要）

| 指标类别 | 具体指标 | 监控意义 |
| --- | --- | --- |
| **显存使用** | GPU Memory Used / Total | 防止 OOM，优化 batch size |
| **显存利用率** | GPU Memory Utilization % | 判断显存是否成为瓶颈 |
| **GPU 算力利用率** | GPU Compute Utilization % | 判断 GPU 是否满载 |
| **KV Cache 使用率** | `vllm:kv_cache_usage_perc` | 最关键指标，决定并发能力 |
| **温度/功耗** | GPU Temperature / Power Draw | 硬件健康状态 |

### 2\. vLLM 服务指标（通过 /metrics 端点）

**延迟指标** ：

- **TTFT** (Time To First Token): 首 token 时间 = 排队时间 + 预填充时间
- **ITL** (Inter-Token Latency): token 间延迟，反映解码阶段性能
- **E2E Latency**: 端到端总延迟

**吞吐量指标** ：

- `vllm:prompt_tokens_total`: 输入 token 总数
- `vllm:generation_tokens_total`: 生成 token 总数
- Token 吞吐量 (tokens/s)

**服务健康指标** ：

- `vllm:num_requests_running`: 正在运行的请求数（当前 batch size）
- `vllm:num_requests_waiting`: 等待队列深度
- `vllm:num_preemptions_total`: 请求被抢占次数（内存压力指标）

### 3\. 系统资源指标

- **CPU 使用率**: 处理前后置任务（序列化、日志）
- **内存使用率**: 主机内存情况
- **网络带宽**: 数据传输瓶颈
- **磁盘 I/O**: 日志写入、模型加载

## 二、方便的监控方案

### 方案 1: 快速命令行监控（适合临时测试）

**终端 1 - 启动 vLLM 服务** ：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

**终端 2 - 实时监控 GPU** ：

```bash
# 基础监控（1秒刷新）
watch -n 1 nvidia-smi

# 或更详细的监控
nvidia-smi dmon -s pucm -d 1
# 显示：功耗、温度、利用率、显存、编码器/解码器状态
```

**终端 3 - 监控 vLLM 指标** ：

```bash
# 查看当前指标
curl http://localhost:8000/metrics | grep -E "(num_requests|kv_cache|latency)"

# 实时监控关键指标
watch -n 1 'curl -s http://localhost:8000/metrics | grep -E "vllm:num_requests_running|vllm:num_requests_waiting|vllm:kv_cache_usage_perc"'
```

**终端 4 - 运行 benchmark** ：

```bash
vllm bench serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host localhost \
    --port 8000 \
    --num-prompts 1000 \
    --max-concurrency 32 \
    --random-input-len 1024 \
    --random-output-len 128
```

### 方案 2: nvitop 交互式监控（推荐）

`nvitop` 是增强版的 `nvidia-smi` ，支持交互式查看和导出到 Prometheus ：

```bash
# 安装
pip install nvitop

# 交互式监控（类似 htop）
nvitop

# 仅监控特定 GPU
nvitop --gpu 0

# 导出为 Prometheus 格式（用于长期监控）
nvitop-exporter --port 9100
```

### 方案 3: Prometheus + Grafana 完整监控栈（生产环境）

这是业界标准方案，支持历史数据存储和可视化 ：

**docker-compose.yml** ：

```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: vllm serve meta-llama/Llama-3.1-8B-Instruct
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"

  dcgm-exporter:
    image: nvidia/dcgm-exporter:latest
    ports:
      - "9400:9400"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
```

**prometheus.yml** ：

```yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:8000']
    metrics_path: '/metrics'
  
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
  
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

**启动监控** ：

```bash
docker-compose up -d
# 访问 Grafana: http://localhost:3000 (admin/admin)
# 访问 Prometheus: http://localhost:9090
```

### 方案 4: Python 脚本自动化监控

适合集成到测试流程中：

```python
import requests
import time
import subprocess
import json
from datetime import datetime

class VLLMMonitor:
    def __init__(self, vllm_url="http://localhost:8000"):
        self.vllm_url = vllm_url
        self.metrics_url = f"{vllm_url}/metrics"
    
    def get_vllm_metrics(self):
        """获取 vLLM 服务指标"""
        try:
            resp = requests.get(self.metrics_url, timeout=5)
            metrics = {}
            for line in resp.text.split('\n'):
                if line.startswith('vllm:num_requests_running'):
                    metrics['running'] = float(line.split()[-1])
                elif line.startswith('vllm:num_requests_waiting'):
                    metrics['waiting'] = float(line.split()[-1])
                elif line.startswith('vllm:kv_cache_usage_perc'):
                    metrics['kv_cache_usage'] = float(line.split()[-1]) * 100
            return metrics
        except Exception as e:
            print(f"Error fetching vLLM metrics: {e}")
            return {}
    
    def get_gpu_metrics(self):
        """获取 GPU 指标"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            util, mem_used, mem_total, temp, power = result.stdout.strip().split(', ')
            return {
                'gpu_utilization': float(util),
                'memory_used_mb': float(mem_used),
                'memory_total_mb': float(mem_total),
                'memory_percent': float(mem_used) / float(mem_total) * 100,
                'temperature': float(temp),
                'power_w': float(power)
            }
        except Exception as e:
            print(f"Error fetching GPU metrics: {e}")
            return {}
    
    def monitor(self, duration_seconds=300, interval=5):
        """持续监控并输出"""
        print(f"开始监控 {duration_seconds} 秒...")
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vllm_m = self.get_vllm_metrics()
            gpu_m = self.get_gpu_metrics()
            
            print(f"[{timestamp}] "
                  f"Running: {vllm_m.get('running', 0):.0f} | "
                  f"Waiting: {vllm_m.get('waiting', 0):.0f} | "
                  f"KV Cache: {vllm_m.get('kv_cache_usage', 0):.1f}% | "
                  f"GPU: {gpu_m.get('gpu_utilization', 0):.1f}% | "
                  f"VRAM: {gpu_m.get('memory_percent', 0):.1f}% | "
                  f"Temp: {gpu_m.get('temperature', 0):.1f}°C")
            
            time.sleep(interval)

# 使用示例
if __name__ == "__main__":
    monitor = VLLMMonitor()
    monitor.monitor(duration_seconds=300, interval=5)  # 监控5分钟，每5秒采样
```

## 三、关键监控策略与调优建议

### 1\. 监控重点

**判断服务器是否饱和** ：

- 如果 `num_requests_waiting` > 0 且持续增加 → **服务器过载，需要扩容或降低并发**
- 如果 `kv_cache_usage_perc` > 90% → **显存压力，需要减小 batch size 或使用量化**
- 如果 `num_preemptions_total` 持续增加 → **严重内存抖动，请求被频繁抢占**

### 2\. 性能瓶颈诊断

| 症状 | 可能原因 | 解决方案 |
| --- | --- | --- |
| TTFT 高，ITL 正常 | 排队延迟或长 prompt 预填充 | 增加 GPU 或减少并发 |
| TTFT 正常，ITL 高 | 解码阶段带宽瓶颈 | 启用投机解码或降低 batch size |
| 两者都高 | 全面过载 | 扩容或优化模型配置 |

### 3\. 实用监控命令组合

```bash
# 一键监控脚本（保存为 monitor.sh）
#!/bin/bash

echo "=== vLLM Benchmark Monitor ==="
echo "Time: $(date)"
echo ""

# 1. GPU 状态
echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv

# 2. vLLM 关键指标
echo ""
echo "--- vLLM Metrics ---"
curl -s http://localhost:8000/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|time_to_first_token|inter_token_latency)" | head -20

# 3. 系统资源
echo ""
echo "--- System Resources ---"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
```

## 四、推荐监控工具对比

| 工具 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| `nvidia-smi` | 快速检查 | 无需安装，随时可用 | 信息有限，无历史数据 |
| `nvitop` | 交互式监控 | 界面友好，支持导出 | 需要额外安装 |
| `Prometheus+Grafana` | 生产环境 | 功能完整，可告警 | 部署复杂 |
| `dcgm-exporter` | 企业级 GPU 监控 | NVIDIA 官方，指标丰富 | 需要 DCGM 支持 |
| 自定义脚本 | 自动化测试 | 灵活，可集成 | 需要开发维护 |

**建议** ：

- **开发/测试阶段** ：使用 `nvitop` + 自定义脚本
- **生产环境** ：使用 `Prometheus + Grafana + DCGM` 完整方案
- **快速验证** ：使用 `nvidia-smi dmon` 或 `watch` 命令

  