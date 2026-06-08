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
![[Pasted image 20260608105457.png]]
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


## headroom 压缩
Headroom 的压缩不是「一刀切」，而是先识别内容类型，再上对应的压缩器。这个设计思路我觉得挺对的，JSON 的压缩策略和代码不应该一样，日志和图片更不应该一样。
**1. SmartCrusher**
专门对付 JSON 和结构化工具输出。比如一次代码搜索返回了 100 个匹配结果，每个都带着一堆元数据。SmartCrusher 不是简单截断，而是统计哪些字段有信息量、哪些是重复的模板字段，然后只保留关键的。实测能压掉 70-90%。
**2. CodeCompressor**
AST 感知的代码压缩。用 tree-sitter 解析代码的抽象语法树，知道哪些是 import、哪些是函数签名、哪些是实现细节。保留结构，精简实现。支持 Python、JS、Go、Rust、Java、C++。
**3. Kompress-base**
通用文本压缩。这是 Headroom 团队自己训的一个模型，挂在 HuggingFace 上（ModernBERT 架构）。专门针对 Agent 工作流中产生的文本，日志、错误信息、RAG 检索片段，做了训练。
**4. ImageCompressor**
图片压缩，ML 路由判断要不要压 + OCR 提取关键文字。40-90% 的缩减。
**5. CacheAligner**
这个想得比较细。Anthropic 和 OpenAI 的 API 都有 prompt cache 机制，但如果你的 prompt 结构不稳定，比如每次请求的消息顺序稍有不同，缓存就命中不了。CacheAligner 专门做前缀对齐，让你的压缩策略不破坏缓存命中率。
**6. CCR（Compress-Cache-Retrieve）**
这可能是整个项目最有意思的设计。
大多数压缩方案是不可逆的，压了就压了，原始信息丢了。如果 LLM 真的需要用某段被压掉的细节，你没法给它。
CCR 的思路是：压缩后的内容发给 LLM，**原始内容存在本地**。同时往压缩后的消息里注入一个 `headroom_retrieve` 工具，LLM 如果觉得某处信息不够，可以主动调这个工具把原文拉回来。

 headroom proxy   --host 0.0.0.0  --port 8787   --backend anyllm   --anyllm-provider openai   --openai-api-url http://192.168.0.181:4000/v1

```python
from openai import Op
enAI

client = OpenAI(api_key="vllm-is-awesome", base_url="http://127.0.0.1:8787/v1")
#client = OpenAI(api_key="vllm-is-awesome",base_url="http://192.168.0.181:8787/v1")

stream = client.chat.completions.create(
    model="qwq-32b",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.usage:
        final_usage = chunk.usage
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print("\nUsage:", final_usage)
```

完整链路总结
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
### 为什么 Proxy 模式不支持强制压缩策略
Headroom 的设计哲学是 **"智能、安全地压缩"**：
- 自动识别可压缩内容（代码、JSON、日志、工具输出）
- 保护自然语言语义（不压缩 user 消息、推理过程）
- 通过 `--learn` 让系统从流量中学习模式
强制压缩自然语言（如病历文本）可能导致**关键医学信息丢失**，所以 Proxy 模式默认不提供这个开关。

Headroom 的压缩策略是**内容类型感知**的，主要针对以下几类数据：

| 内容类型                    | 压缩效果      | 你的场景         |
| :---------------------- | :-------- | :----------- |
| **JSON 数据**             | 59-90%    | ❌ 不是 JSON    |
| **代码/日志**               | 31-94%    | ❌ 不是代码/日志    |
| **工具输出 (tool results)** | 高压缩       | ❌ 没有 tool 角色 |
| **RAG 片段**              | 中等压缩      | ❌ 不是 RAG     |
| **自然语言文本**              | **默认不压缩** | ✅ 你的病历文本属于此类 |

Headroom 的设计理念是 **"protect the latest, shave the bulky"** —— 保护最新的用户消息和推理过程，只压缩累积的工具输出和过时的上下文。
**跟其他方案比**
||压缩范围|部署方式|本地|可逆|
|--|--|--|-|-|
|**Headroom**|全部上下文|Proxy / Library / MCP|✅|✅|
|RTK|CLI 命令输出|CLI Wrapper|✅|❌|
|lean-ctx|CLI/MCP 工具|CLI Wrapper|✅|❌|
|Compresr / Token Co.|文本|云端 API|❌|❌|
|OpenAI 原生 Compaction|对话历史|提供商内置|❌|❌|


最大的差别在两点：
一是 **覆盖范围**。RTK 和 lean-ctx 只压 CLI 输出，Cloud 方案只压文本，OpenAI 原生只压对话历史。Headroom 什么内容类型都压，JSON、代码、图片、日志。
二是 **可逆性**。只有 Headroom 做了 CCR，其他方案压了就没了。如果你在意「LLM 需要的时候能回到原文」，这是唯一的选项。

### 强制压缩策略
```python
import headroom
messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': '入院记录：###\n患者性别=男 患者年龄=75\n姓名：季章明 职    业：工人 \r\n性别：男 工作单位：浙江省金华市永康市上西季村 \r\n年龄：75岁  住    址：浙江省金华市永康市上西季村 \r\n婚姻：已婚 供史者：家属  可靠 \r\n出生地：浙江省金华市永康市上西季村 入院时间：2024年08月19日 08时31分 \r\n民族：汉族 记录时间：2024年08月19日 16时26分 \r\n病     史\r\n主诉：发现肝占位2天\r\n现病史：患者2天前体检就诊于永康市第一人民医院查afp:10.10(ng/ml),腹部ct：2024-08-17永康人民医院增强ct：肝左内叶稍低密度灶；建议进一步检查。n###'}]
    
# 强制压缩所有消息（包括自然语言）
result = headroom.compress(
    messages=messages,
    model="qwq-32b",  # 或 "claude-sonnet" 等
    target_ratio=0.5,        # 目标压缩到 50%
    protect_recent=0,        # 不保护最近消息，全部压缩
    compress_user_messages=True,  # 强制压缩 user 消息（默认不压缩）
)

print(f"压缩前 tokens: {result.tokens_before}")
print(f"压缩后 tokens: {result.tokens_after}")
print(f"节省: {result.tokens_saved}")

compressed_messages = result.messages
print("compressed_messages:", compressed_messages)
```