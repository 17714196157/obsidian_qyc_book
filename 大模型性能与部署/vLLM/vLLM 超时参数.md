---
tags:
  - vllm
  - 性能调优
---

vLLM 的超时机制涉及多个层面，**没有单一的 `--timeout` 启动参数**，而是通过环境变量和客户端配置共同控制：

### 1. 引擎内部超时（环境变量）

| 环境变量 | 默认值 | 说明 | 超时后行为 |
|----------|--------|------|------------|
| **`VLLM_ENGINE_ITERATION_TIMEOUT_S`** | **60 秒** | 引擎单次迭代（step）的最大执行时间   | **引擎崩溃**，抛出 `Engine iteration timed out. This should never happen!` 错误   |
| **`VLLM_ENGINE_READY_TIMEOUT_S`** | **600 秒** (10分钟) | 引擎启动时等待 core 就绪的时间  | 启动失败 |
| **`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`** | 依赖具体版本 | `execute_model` RPC 调用超时（多进程执行器） | 任务失败 |
| **`VLLM_RPC_TIMEOUT`** | - | RPC 通信超时（微秒或毫秒级） | 通信失败 |
| **`VLLM_IMAGE_FETCH_TIMEOUT`** | **5 秒** | 多模态模型下载图片的超时时间  | 图片加载失败 |

### 2. 客户端超时（HTTP/Client 级别）

**OpenAI Python 客户端：**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.172:8102/v1/",
    api_key="not-needed",
    timeout=120.0  # 客户端超时时间（秒）
)
```
OpenAI 客户端超时后**直接报错，不会自动重试**

```python
# 如果需要自动重试
from openai import OpenAI, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential
client = OpenAI(
    base_url="http://192.168.0.172:8102/v1/",
    timeout=60.0
)
@retry(
    stop=stop_after_attempt(3),           # 最多重试3次
    wait=wait_exponential(multiplier=1, min=4, max=10),  # 指数退避
    retry=lambda e: isinstance(e, APITimeoutError)  # 只有超时错误才重试
)
def chat_with_retry():
    return client.chat.completions.create(
        model="your-model",
        messages=[{"role": "user", "content": "Hello"}]
    )
try:
    response = chat_with_retry()
except APITimeoutError:
    print("重试3次后仍然超时")
```
**HTTP 请求超时（curl/httpx）：**
```bash
# curl 连接超时和读取超时
curl --connect-timeout 10 --max-time 120 \
  http://192.168.0.172:8102/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## 超时后的行为
### 情况 1：客户端超时（最常见）

- **现象**：客户端主动断开连接
- **服务端行为**：vLLM 会继续生成直到完成，但**结果会被丢弃** 
- **资源浪费**：GPU 仍在为该请求计算，但客户端已收不到结果
- **建议**：对于长文本生成，建议设置较大的客户端超时，或使用流式输出（streaming）

### 情况 2：`VLLM_ENGINE_ITERATION_TIMEOUT_S` 超时（严重）

- **现象**：日志显示 `Engine iteration timed out. This should never happen!`  
- **原因**：单次推理迭代（**一次 forward pass（前向传播）= 一次 prefill 或一次 decode step**）超过 60 秒未完成，通常由 NCCL 通信阻塞、GPU  hang 或极端负载导致
- **后果**：
  - 引擎标记为 `errored` 状态
  - 触发 `AsyncEngineDeadError`
  - **后台循环（background loop）停止**
  - **所有正在处理的请求失败**
  - 服务可能变得不可用，需要重启 

### 情况 3：队列等待超时
- 当 `--max-num-seqs` 达到上限时，新请求进入等待队列
- 如果等待时间过长，客户端可能先超时断开
- 服务端会继续处理队列中的请求


## 生产环境建议配置
```bash
# 对于长文本或复杂模型，增加引擎迭代超时
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300  # 5分钟，默认60秒可能太短

# 如果是多模态模型，增加图片下载超时
export VLLM_IMAGE_FETCH_TIMEOUT=30

# 启动服务时限制并发和长度，避免单次迭代过长
vllm serve your-model \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85
```

**关键区别**：
- 客户端超时 → 仅影响该客户端，服务端继续计算（浪费资源）
- 引擎迭代超时 → 影响整个服务，会导致引擎崩溃（严重错误）

建议根据模型推理速度合理设置 `VLLM_ENGINE_ITERATION_TIMEOUT_S`，对于大模型或长序列，通常需要设置为 300-1800 秒  。