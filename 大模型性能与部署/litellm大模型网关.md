## litellm
```
(tt) root@maizi:/home/maizidata/基础知识练习杂记/LLM的并行测试# cat litellm_config.yaml 
model_list:
  - model_name: qwq-32b
    litellm_params:
      model:  openai/qwq-32b      # ← 必须加 openai/ 前缀
      api_base: http://192.168.0.180:8102/v1
      api_key: vllm-is-awesome
```


**启动 litellm --config litellm_config.yaml --host 0.0.0.0 --port 4000 **
![[file-20260709175737401.png]]
```
from openai import OpenAI
client = OpenAI(
    base_url="http://192.168.0.181:4000",
    api_key="vllm-is-awesome"
)
response = client.chat.completions.create(
    model="qwq-32b",  # Must match --served-model-name in vLLM
    messages=[{"role": "user", "content": "你好,你是谁!"}]
)
print(response.choices[0].message.content)
```

**完整请求链路总结** 
```
你的客户端 
    ↓ 请求 192.168.0.181:8787/v1
Headroom Proxy (8787) 
    ↓ 转发到 192.168.0.181:4000/v1
LiteLLM (4000) 
    ↓ 转发到 192.168.0.180:8102/v1
vLLM (8102) 
    ↓ 返回推理结果
```



### 例子2） 请求vllm的8102端口服务， litellm转到阿里云的api服务

```
model_list:
  # 本地 VLLM 服务 - 端口 8102 转 阿里云 DashScope 服务
  - model_name: qwq-32b
    litellm_params:
      model: openai/deepseek-v4-flash
      api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: sk-d98a7434af1f4641921b8af02e175499
      timeout: 1200           # 网络请求超时 20分钟
      stream_timeout: 1200    # 流式网络请求超时 20分钟
    model_info:
      mode: completion

  - model_name: qwen2.5-14b-instruct
    litellm_params:
      model: openai/deepseek-v4-flash
      api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: sk-d98a7434af1f4641921b8af02e175499
      timeout: 1200           # 网络请求超时 20分钟
      stream_timeout: 1200    # 流式网络请求超时 20分钟
    model_info:
      mode: completion

router_settings:
  timeout: 1200              # Router层总超时 20分钟（可选，但建议保留）
  retry_after: 0             # 建议关闭重试，否则20分钟会被重试占用

```
**只配置 `router_settings.timeout` 的核心问题**：
> 它只能保证 Router 层等待20分钟，但**实际发往模型供应商的 HTTP 连接可能早在几十秒后就超时断开了**，导致流式请求在收到完整响应前就被中断。
必须同时配置模型参数中的 `timeout` 和 `stream_timeout` 才能让底层的 HTTP 客户端真正等到20分钟


litellm --config config.yaml --port 8102
```
http://192.168.0.181:8102/v1/chat/completions
{
    "model": "qwq-32b",
    "messages": [
        {
            "role": "user",
            "content": "你好，介绍一下 vLLM 的并发能力"
        }
    ],
    "max_tokens": 100,
    "temperature": 0.7
}
```