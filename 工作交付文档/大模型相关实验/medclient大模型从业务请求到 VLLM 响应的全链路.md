### 展示从业务请求到 VLLM 响应的全链路日志记录逻辑：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              业务流程图：MedClient ↔ VLLM                          │
│                         通过 HTTP_X_REQUEST_ID 实现全链路追踪                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────────────────────────────┐     ┌─────────────────┐
│   外部客户端   │────▶│         MedClient (Django)          │────▶│   VLLM 服务      │
│  (医院/前端)  │     │    ┌─────────────────────────────┐    │     │  (大模型推理)    │
└─────────────┘     │    │    LogMiddleware 中间件      │    │     └─────────────────┘
                    │    │                             │    │              │
                    │    │  ① 接收业务请求                │    │              │
                    │    │     POST /opt/OptXihua        │    │              │
                    │    │                             │    │              │
                    │    │  ② 生成 X-Request-Id          │    │              │
                    │    │     = medicalRecordId(请求参数)   │    │              │
                    │    │       + work(请求参数)         │    │              │
                    │    │       + url                   │    │              │
                    │    │       + timestamp             │    │              │
                    │    │     例: "2023477759-手术另编-   │    │              │
                    │    │          /opt/OptXihua-1713..."│    │              │
                    │    │                             │    │              │
                    │    │  ③ 记录业务请求日志             │    │              │
                    │    │     [X-Request-Id] REQ ...    │    │              │
                    │    │                             │    │              │
                    │    │  ④ 拼接 Prompt                │    │              │
                    │    │     医学提示词 + 手术记录        │    │              │
                    │    │                             │    │              │
                    │    │  ⑤ 调用 VLLM                  │    │              │
                    │    │     HTTP Header:              │────┘              │
                    │    │     X-Request-Id: <生成的ID>   │◄──────────────────┘
                    │    │                             │      ⑥ VLLM 接收请求
                    │    │  ⑩ 接收 VLLM 响应              │◄──────────────────┐
                    │    │                             │      ⑦ 记录 VLLM 日志
                    │    │  ⑪ 记录业务响应日志            │      (输入/输出关联ID)
                    │    │     [X-Request-Id] RES ...    │                    │
                    │    │                             │      ⑧ 大模型推理     │
                    │    │  ⑫ 返回客户端                 │      ⑨ 返回结果      │
                    │    └─────────────────────────────┘◄────────────────────┘
                    │              ▲                           │
                    │              │                           │
                    └──────────────┼───────────────────────────┘
                                   │
                    ┌──────────────┴───────────────────────────┐
                    │              日志系统                        │
                    │  ┌─────────────────────────────────────┐   │
                    │  │  MedClient 日志                      │   │
                    │  │  [2023477759-...] REQ POST /opt/...  │   │
                    │  │  [2023477759-...] RES 200 {...}      │   │
                    │  └─────────────────────────────────────┘   │
                    │  ┌─────────────────────────────────────┐   │
                    │  │  VLLM 日志                            │   │
                    │  │  [2023477759-...] 输入: prompt...    │   │
                    │  │  [2023477759-...] 输出: 推理结果...   │   │
                    │  └─────────────────────────────────────┘   │
                    │                                            │
                    │  通过相同 X-Request-Id 关联两端日志          │
                    └────────────────────────────────────────────┘
```

---

## 简化版流程（时序图）

```
时间轴 ─────────────────────────────────────────────────────────────►

外部客户端        MedClient(Django)              VLLM服务
    │                    │                         │
    │ ① POST /opt/OptXihua│                         │
    │ ──────────────────▶│                         │
    │                    │                         │
    │                    │ ② 生成 X-Request-Id      │
    │                    │    "2023477759-手术另编- │
    │                    │     /opt/OptXihua-..."   │
    │                    │                         │
    │                    │ ③ 得到业务请求麻溜        │
    │                    │ ───────────────────────┐ │
    │                    │                        │ │
    │                    │ ④ 拼接医学Prompt        │ │
    │                    │                        │ │
    │                    │ ⑤ HTTP请求 VLLM       │ │
    │                    │    Header: X-Request-Id│ │
    │                    │ ──────────────────────▶│ │
    │                    │                        │ │
    │                    │                        │⑥ 接收请求
    │                    │                        │⑦ 打印VLLM输入请求
    │                    │                        │  [X-Request-Id] 输入:...
    │                    │                        │⑧ 大模型推理
    │                    │                        │⑨ 返回推理结果
    │                    │                        │⑩ 写日志：VLLM输入输出日志
    │                    │                        │  [X-Request-Id] 输出:...
    │                    │⑪ 接收VLLM响应 ◀────────│ │
    │                    │                        │ │
    │                    │⑫ 写日志：业务请求 + 业务响应 ──────┤ │
    │                    │  [X-Request-Id] RES ...│ │
    │                    │                        │ │
    │⑬ 返回结果 ◀────────│                        │ │
    │                    │                        │ │
```

---

## 关键数据流向

| 阶段  | 位置               | X-Request-Id 生成/传递                              | 日志记录      |
| :-- | :--------------- | :---------------------------------------------- | :-------- |
| 入口  | MedClient 中间件    | `md5(medicalRecordId + work + url + timestamp)` | ✅ 业务请求    |
| 出口  | MedClient → VLLM | HTTP Header `X-Request-Id`                      | -         |
| 入口  | VLLM 服务          | 从 Header 读取                                     | ✅ VLLM 输入 |
| 出口  | VLLM 服务          | 相同 ID 关联                                        | ✅ VLLM 输出 |
| 出口  | MedClient → 客户端  | Response Header 带回                              | ✅ 业务响应    |
## 附件
### 附件1） vllm 中日志格式
vllm日志中记录的请求和响应， 通过 X-Request-Id 关联
![[file-20260706150357623.png]]
### 附件2）medclient 中日志格式
![[file-20260706150357653.png]]


### 附件3）django中如何通过中间件记录日志

```python
# utils/log_middleware.py
class LogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 记录请求
        body = request.body.decode('utf-8', errors='ignore') if request.body else ''
        try:
            response = self.get_response(request)
            # 记录正常响应    
            ## 记录响应
            content = response.content.decode('utf-8', errors='ignore') if response.content else ''    
        except Exception as e:
            # 记录异常
            content = f"RES EXCEPTION {type(e).__name__}: {str(e)}"    

        finally:
            # 记录响应
            logger.debug(f"REQ {request.path}$$\n\n{body}\n\n$$RES$${response.status_code}$$\n\n{content}\n\n$$END")

            return response
```

```
# settings.py
MIDDLEWARE = [
    'middleware.log_middleware.LogMiddleware',  # 放最前面
    # ... 其他中间件
]
```
![[django中间件.png]]



### 附件4）vllm如何实现记录请求响应日志

修改源代码：
在你的 `serving_chat.py` 文件中，找到 `chat_completion_full_generator` 方法的末尾（大约第 686 行附近），在 `return response` 之前添加：

```python
vi /root/miniconda3/envs/sglong/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py

# 在构建好响应对象 'response' 后添加
print(f"Request Prompt:{request_id} {request.messages}") # 打印请求的完整消息
print(f"Response Text:{request_id} {response.choices[0].message.content}") # 打印模型生成的文本
print(f"Response Text:{request_id} {usage}")

```

![[6e1562548d4f69a84c177e4a70345e7f_MD5.png]]

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