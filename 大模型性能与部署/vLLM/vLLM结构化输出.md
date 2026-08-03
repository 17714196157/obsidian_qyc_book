---
title: vLLM结构化输出：3个生产坑
source: https://mp.weixin.qq.com/s/NfZHSzSSU0hUk4dpI2j3DA
author:
published:
created: 2026-08-03
description: 读完这篇你能带走什么：用 guided decoding 让模型只吐合法 JSON，并避开延迟、掉质量、认知三
tags:
  - clippings
  - vllm
  - guided-decoding
  - 结构化输出
  - xgrammar
---
## 概述

> [!abstract] 读完这篇你能带走什么
> 用 guided decoding 让模型只吐合法 JSON，并避开延迟、掉质量、认知三个坑。
> 
> **适用场景**：模型输出要喂给下游代码解析（Agent 工具调用、数据抽取、API 返回值）
> **不适用**：纯聊天、创意写作等自由文本

我们的 Agent 上线第一周，工具调用每天崩几十次。根因不是模型笨，是 JSON 解析失败——模型偶尔在 JSON 外包一层 ` ```json ` 代码块、少个引号、或者输出太长被截断。我们在 prompt 里写了"只返回 JSON"，但**"写"不等于"保证"**。

后来我们用 vLLM 的结构化输出（guided decoding）把解析失败率从 1% 压到 0。但落地时踩了三个坑，竞品的入门教程一个都没提。

---

## 一、为什么"让模型返回 JSON"不够

靠 prompt 约束格式，本质是**"请求"**，不是"保证"。模型按概率采样 token，没有任何机制阻止它吐出破坏 schema 的 token。生产流量上，纯 prompt 的解析失败率是 **5%-30%**（来源：Local AI Master 实测）。你几千次调用/天，1% 失败就是几十个坏请求，下游 Agent 根本不关心模型"大部分时候是对的"。

结构化输出的原理很简单：每一步解码前，把违反结构的 token 概率设为负无穷，模型只能从合法的 token 里采样。结构由状态机（JSON 用下推自动机）跟踪。结果是**从构造上就合法**，不是靠运气。

这里有个关键认知误区：**JSON mode ≠ schema 约束**。

| 方式 | 保证什么 | 不保证什么 |
|------|---------|-----------|
| **JSON mode** `response_format={"type":"json_object"}` | 语法合法的 JSON | 形状任意——字段名、类型、枚举都不保证 |
| **Schema 约束** `json_schema` | 该有的字段、对的类型、对的枚举值都在 | 语义正确性（见坑 3） |

> [!important] 关键判断
> 如果你的下游按固定字段解析，JSON mode 照样会失败。要的是 `json_schema`，不是 `json_object`。

## 二、怎么落地：一行后端 + 一个 schema

vLLM 0.8.5+ 结构化输出已稳定可用，引导解码默认走 xgrammar 后端（`auto` 模式会自动挑最优）。想显式锁定也可以：

```bash
vllm serve Qwen/Qwen3-8B \
  --guided-decoding-backend xgrammar
```

请求时把 Pydantic 模型直接塞进 `response_format`（OpenAI SDK 1.40+ 支持）：

```python
from openai import OpenAI
import numpy as np
import os
import re
import json
from icecream import ic
from tqdm import tqdm
# api_key =  "sk-d98a7434af1f4641921b8af02e175499"  # 公式key 超哥给的新的
# base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key =  "vllm-is-awesome" # 公式key 超哥给的新的
base_url = "http://192.168.0.180:8102/v1"

client_vllm = OpenAI(
  base_url = base_url,
  api_key = api_key  # required, but unused
)

from pydantic import BaseModel
class Verdict(BaseModel):
    reasoning: str      # 让模型在这里自由推理，不约束格式和内容
    category: str       # 只对最终结论做结构约束

content = """
分类这条工单
"""

# # 方案一（推荐，兼容 vLLM）：获取 JSON Schema 手动传入,需要 Pydantic 版本是 v2， model_json_schema() 是 v2 才有的方法
# schema = Verdict.model_json_schema()
# resp = client_vllm.chat.completions.create(
#     model="qwq-32b",
#     messages=[{"role": "user", "content": content}],
#     response_format={
#         "type": "json_schema",
#         "json_schema": {
#             "name": "Verdict",
#             "schema": schema,
#             "strict": True  # vLLM 开启严格模式
#         }
#     }
# )

# verdict = Verdict.model_validate_json(resp.choices[0].message.content)
# print(verdict.reasoning)
# print(verdict.category)
# print(verdict.confidence)

# 手动定义 JSON Schema（完全独立于 Pydantic 版本）
schema = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "category": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["reasoning", "category", "confidence"],
    "additionalProperties": False   # 禁止额外字段，保持严格
}

resp = client_vllm.chat.completions.create(
    model="qwq-32b",
    messages=[{"role": "user", "content": content}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "Verdict",
            "schema": schema,
            "strict": True   # 如果你 vLLM 版本支持；若报错可删掉
        }
    }
)

# 解析响应：先用 json.loads 解析为字典，再构建 Verdict 对象（兼容 v1/v2）
data = json.loads(resp.choices[0].message.content)
verdict = Verdict(**data)   # 或者 Verdict.parse_obj(data) 也可以
print(verdict)
# print(verdict.reasoning)
# print(verdict.category)
```

输出保证能 `model_validate_json` 通过，不用正则清洗、不用去 markdown fence、不用重试。我们线上把工单分类、实体抽取、工具参数的返回全部换成这个写法，解析失败率从 1% 降到 0。

## 三、3 个生产坑
### 坑 1：延迟税（性能）
guided decoding 不是免费的。每步要多算一次 token 掩码，会吃掉一部分解码吞吐：

| 后端 | 每 token 开销 | 吞吐损耗 |
| --- | --- | --- |
| xgrammar | 0.5-1.5 ms | 5-10% |
| outlines | 1-3 ms | 10-20% |
| lm-format-enforcer | 1-3 ms | 10-20% |

我们实测：默认 xgrammar 在复用 schema 时开销最低（它把语法编译后缓存）。 **别在自由文本端点开结构化输出** ——只对真正要解析的接口开，schema 尽量复用让缓存命中。

### 坑 2：过度约束掉质量（质量）
这是最隐蔽的坑。过早把模型锁死在刚性 schema 里，会切掉它的推理空间，结果是"字段合法但内容胡说"。
正确做法：让模型先在自由文本里推理，只在最终答案上约束。加一个 `reasoning` 字段承载思考：
```python
class Verdict(BaseModel):
    reasoning: str      # 先让模型自由推理
    category: str       # 再约束最终结论
```

结构只约束"答案"，不约束"思考"。我们遇到过一次：把整段输出都塞进严格 schema，分类准确率掉了 8 个点，放开 reasoning 字段后回来。

### 坑 3：JSON mode 认知误区 + 兜底（可靠性）
两个细节：
1. 前面说了，JSON mode 不保证你的形状。确认下游是按字段解析的，必须用 `json_schema`。
2. 结构化输出保证"结构合法"，**不保证"语义正确"**。枚举值合法不代表业务对。极端情况下（`max_tokens` 卡在 JSON 中间、空 enum 导致死循环）仍可崩。保留一层兜底：
```python
import json

def safe_parse(content, model):
    try:
        return model.model_validate_json(content)
    except Exception:
        # 极端兜底：记日志 + 走校验分支
        return None
```
grammar 约束结构，业务规则用代码兜底——**两层都留**，才不会半夜被叫醒。

---

## 结语

> [!quote] 一句话收口
> 结构化输出治的是格式的病，治不了逻辑的病：它**保证形状，不保证正确**。让模型在自由里思考，在约束里作答。

## 你明天就能做的 3 件事
- [ ] **命令**：给 vLLM 启动加 `--guided-decoding-backend xgrammar`，锁定默认后端
- [ ] **改造**：把所有"下游要解析"的返回换成 `response_format=Pydantic模型`，删掉正则清洗代码
- [ ] **检查**：schema 里加 `reasoning` 字段承接推理，并保留 `safe_parse` 兜底函数