## 一）vllm自带并行测试工具
`vllm bench serve`是 vLLM 提供的在线服务吞吐量基准测试工具，用于测试已运行的 vLLM 服务器性能。
##### 使用自定义数据集的方法
自定义数据集需要保存为 **JSONL 格式**（每行一个 JSON 对象），必须包含 `prompt` 字段
基本格式示例 (`data.jsonl`)：
```jsonl
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```
**先启动一个vllm服务**
```bash
# 运行自定义数据集基准测试
vllm bench serve \
    --dataset-name custom \
    --dataset-path /path/to/your/data.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --custom-output-len 256 \
    --num-prompts 100 \
    --max-concurrency 32
```
##### 常用参数说明

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
##### 输出指标说明
运行后会输出以下关键性能指标：
- **Request throughput (req/s)**: 请求吞吐量
- **Input token throughput (tok/s)**: 输入 token 吞吐量
- **Output token throughput (tok/s)**: 输出 token 吞吐量
- **Mean TTFT (ms)**: 首 token 延迟平均值（Time To First Token）
- **Mean TPOT (ms)**: 每输出 token 延迟平均值（Time Per Output Token）
- **Mean ITL (ms)**: token 间延迟平均值（Inter-token Latency）


## 资源监控
##### vLLM 服务指标（通过 `/metrics` 端点 http://localhost:8000/metrics ）
**延迟指标**：
- **TTFT** (Time To First Token): 首 token 时间 = 排队时间 + 预填充时间
- **ITL** (Inter-Token Latency): token 间延迟，反映解码阶段性能
- **E2E Latency**: 端到端总延迟
**吞吐量指标**：
- `vllm:prompt_tokens_total`: 输入 token 总数
- `vllm:generation_tokens_total`: 生成 token 总数
- Token 吞吐量 (tokens/s)
**服务健康指标**：
- `vllm:num_requests_running`: 正在运行的请求数（当前 batch size）
- `vllm:num_requests_waiting`: 等待队列深度
- `vllm:num_preemptions_total`: 请求被抢占次数（内存压力指标）

##### python脚本监控资源情况
![[未命名.base]]
