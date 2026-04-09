---
tags:
  - vllm
---

修改源代码：
在你的 `serving_chat.py` 文件中，找到 `chat_completion_full_generator` 方法的末尾（大约第 686 行附近），在 `return response` 之前添加：

```python
vi /root/miniconda3/envs/sglong/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py

# 在构建好响应对象 'response' 后添加
print(f"Request Prompt:{request_id} {request.messages}") # 打印请求的完整消息
print(f"Response Text:{request_id}{response.choices[0].message.content}") # 打印模型生成的文本
if response.choices and len(response.choices) > 0:
    print(f"[RESPONSE]{response.choices[0].message.content}")
else:
    print("[RESPONSE] No choices in response")
# 强制刷新输出（重要！）
import sys
sys.stdout.flush()
```
![[file-20260409181605217.png]]

``` bash
# 设置环境变量
export VLLM_LOGGING_LEVEL=DEBUG      # 开启DEBUG日志
export VLLM_CONFIGURE_LOGGING=1       # 启用vLLM日志配置
# export VLLM_LOGGING_CONFIG_PATH=/home/qyc/logging_config.json

# 启动服务
vllm serve /home/qyc/bert/Qwen2-0.5B --host 0.0.0.0 --port 8000   --dtype half \
       --enforce-eager \
       --max-num-batched-tokens 8192 \
       --max-num-seqs 4 \
       --enable-chunked-prefill \
       --enable-prefix-caching \
      >> /home/qyc/vllm.log 2>&1 &
```
模拟请求vllm
```
curl --location 'http://192.168.0.172:8000/v1/chat/completions' \
--header 'X-Request-Id: my-unique-request-123' \
--header 'Content-Type: application/json' \
--data '{
    "model": "/home/qyc/bert/Qwen2-0.5B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens":200
}'

响应麻溜：
{

    "id": "chatcmpl-my-unique-request-123",
    "object": "chat.completion",
    "created": 1775728828,
    "model": "/home/qyc/bert/Qwen2-0.5B",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": null,
                "content": "Yes, I see your point. Let me try to think about it.\nhow to prove best of times f(x)=x² for x!=5000, for x in range of 10000, 0<x<10000?\nIt was 1903 when Ford first removed lightning from the wires to reduce arcing and improve lightning shedding. It was 1920 when electrons spilled across prisms and exploded. It was 1930 when a team of engineers at the Pennsylvania Zinc Company at the Marianus Wire Works in Pennsauvillian, Pennsylvania, first experienced what would become known as The Work with the Hateful Apple. Two years after that the First School of Consulting Engineers (FSCE's) was founded as a chartered consulting engineering company serving the electrical industry. On October 5, 1941, the First School of Consulting Engineers' hub became a bronze-t",
                "tool_calls": []
            },
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 19,
        "total_tokens": 219,
        "completion_tokens": 200,
        "prompt_tokens_details": null
    },
    "prompt_logprobs": null
}
```
vllm日志中记录的请求和响应， 通过 X-Request-Id 关联
![[vllm日志中记录的请求和响应效果.png]]