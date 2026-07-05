## litellm
```
(tt) root@maizi:/home/maizidata/基础知识练习杂记/LLM的并行测试# cat litellm_config.yaml 
model_list:
  - model_name: qwq-32b
    litellm_params:
      model:  openai/qwq-32b
      api_base: http://192.168.0.180:8102/v1
      api_key: vllm-is-awesome
```
**启动 litellm --config litellm_config.yaml --host 0.0.0.0 --port 4000 **
![[file-20260705100407817.png]]
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