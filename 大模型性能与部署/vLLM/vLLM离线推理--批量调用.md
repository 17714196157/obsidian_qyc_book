---
title: vLLM 离线推理很好用，不要再批量调用在线服务了
source: https://mp.weixin.qq.com/s/acviYy8i_1hWja57_VbCnQ
author: 有点文艺细菌的码
published: 2026-04-20
created: 2026-04-23
tags:
  - LLM/vLLM
  - inference/offline
  - clippings
---

# vLLM 离线推理

> 说实话，我以前一直搞错了 vLLM 的用法——以为 vLLM 就是 `vllm serve` 起个服务再调 API？那你可能错过了一半的性能。

## 核心痛点

手头有 10 万条数据要做文本分类，或者要把一批用户评论生成摘要。第一反应通常是：

1. `vllm serve` 起一个 OpenAI 兼容的服务
2. 写个 Python 脚本循环调 API
3. 等着……慢慢等……

在线服务（Online Serving）是为实时交互设计的，需要处理并发、管理连接、做请求排队……但当你只是要跑一批数据的时候，这些"服务"开销全是浪费。

> [!important] 结论
> 这时候你需要的，是 vLLM 的**离线推理（Offline Inference）**。

## 什么是离线推理？

> [!info] 一句话定义
> 离线推理就是**直接用 Python 调模型，不经过网络服务层**。

vLLM 的离线推理入口是 `LLM` 类，定义在 `vllm/entrypoints/llm.py`：

> [!note] 源码注释
> `LLM` 类是给离线推理用的，在线服务请用 `AsyncLLMEngine`。

### 最简单的用法（3 行代码）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8))
```

没有 HTTP 服务器，没有 API 调用，没有网络延迟。直接在进程内把活干完。

![[file-20260508235253540.webp]]

## 离线 vs 在线：核心区别

| 维度 | 离线推理（`LLM` 类） | 在线服务（`vllm serve`） |
| --- | --- | --- |
| 调用方式 | Python 函数调用 | HTTP API（OpenAI 兼容） |
| 网络开销 | 零 | 每次请求都有序列化/反序列化 |
| 批量处理 | 原生支持，自动 batch | 需要客户端自己管理并发 |
| 延迟 | 仅模型推理时间 | 模型推理 + 网络传输 + 请求排队 |
| 吞吐量 | 高（GPU 利用率拉满） | 受限于服务层调度 |
| 适用场景 | 批量数据处理 | 实时对话、API 服务 |
| 部署复杂度 | 一个 Python 脚本 | 需要管理服务进程、端口、监控 |

> [!tip] 核心结论
> 如果你不是在做 ChatGPT 那种实时对话产品，**离线推理的性价比远高于在线服务**。

## 离线推理快在哪？

### 1. 自动批量调度——你只管丢数据

`generate` 方法的源码注释：

> "automatically batches"——**自动批处理**！你把 10 万条 prompt 丢进去，vLLM 内部根据 GPU 显存自动决定每次 batch 多少条，你完全不用操心。

而用在线服务呢？你得自己写并发逻辑，控制 `max_concurrency`，还得处理限流和超时。

### 2. 显存利用率拉满

离线推理默认 `gpu_memory_utilization=0.9`，也就是 90% 的 GPU 显存都给 KV Cache 用。在线服务因为要预留资源处理突发请求，实际利用率通常只有 60-70%。

### 3. 零网络开销

| 类型 | 请求链路 |
| --- | --- |
| **在线服务** | 客户端 → HTTP/JSON 序列化 → 网络传输 → 服务端反序列化 → 推理 → 序列化 → 网络传输 → 客户端反序列化 |
| **离线推理** | Python 列表 → 推理 → Python 列表 |

对于本地批量任务，HTTP 那一圈折腾完全是白费。

## 离线推理的 6 大使用场景

### 场景一：批量文本生成

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-72B-Instruct")
prompts = [f"请为以下评论生成摘要：\n{review}" for review in reviews]
outputs = llm.generate(prompts, SamplingParams(max_tokens=200, temperature=0.3))
```

### 场景二：文本分类

```python
llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling", enforce_eager=True)
outputs = llm.classify(prompts)
for prompt, output in zip(prompts, outputs):
    probs = output.outputs.probs  # 直接拿到分类概率
```

> [!warning] 注意
> 分类模型不需要自回归生成，用的是 `runner="pooling"`，效率更高。

### 场景三：文本嵌入（Embedding）

```python
llm = LLM(model="intfloat/e5-small", runner="pooling", enforce_eager=True)
outputs = llm.embed(prompts)
for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding  # 拿到向量表示
```

### 场景四：Reranking / 评分

```python
llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling", enforce_eager=True)
outputs = llm.score(query, documents)  # 输出相关性分数
```

### 场景五：Reward Model 评分

```python
llm = LLM(model="internlm/internlm2-1_8b-reward", runner="pooling", enforce_eager=True)
outputs = llm.reward(prompts)  # 输出 reward 分数
```

### 场景六：Chat 批量推理

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "Write an essay about higher education."},
]
conversations = [conversation for _ in range(10)]
outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
```

> [!tip] 提示
> `llm.chat()` 会自动应用模型的 chat template，不用手动拼 prompt。

## 原理：vLLM 是怎么做到高效的？

### 核心：PagedAttention + 连续批处理

vLLM 的核心优化是 **PagedAttention（分页注意力机制）**。它把 KV Cache 像操作系统管理内存页一样管理，避免了传统推理框架中 KV Cache 的显存碎片问题。

在离线推理中，这体现为 `_run_engine` 循环中的**连续批处理（Continuous Batching）**：

1. 所有 prompt 一次性加入引擎
2. 每个 `step()` 调用中，调度器根据当前显存状态决定哪些请求可以并行
3. 已完成的请求立刻释放显存，新请求立即补上
4. 循环直到所有请求完成

这意味着：你丢 10 万条 prompt 进去，vLLM 不会等你全部处理完才返回。每条完成的结果都会即时收集。

### 新特性：enqueue + wait_for_completion

```python
# 先入队，不等待
request_ids = llm.enqueue(prompts, sampling_params)
# 可以做别的事情……
# 等待完成
outputs = llm.wait_for_completion()
```

给离线推理增加了灵活性——可以在推理的同时做数据预处理。

### CPU Offload：小显存跑大模型

> [!tip] CPU Offload
> 你有 24GB 的 GPU，设 `cpu_offload_gb=10`，等于变出一个 34GB 的"虚拟 GPU"，能跑原来放不下的大模型。对于 13B 的模型（BF16 需要至少 26GB），这个功能简直是救命。

## 离线推理的 5 种实现方式

| 方式 | 说明 | 适用场景 |
| --- | --- | --- |
| **`LLM` 类** | 3 行代码，零配置 | 90% 的场景，最推荐 |
| **`LLMEngine`** | 更细粒度的控制 | 需要精细控制推理流程 |
| **数据并行（Data Parallel）** | 多 GPU 并行处理不同分片 | 多 GPU 环境 |
| **Ray Data 集群** | 生产级大规模批处理 | 自动分片、负载均衡、容错重试 |
| **分离式预填充** | 预填充和解码拆到不同 GPU | 超长 prompt 场景 |

## 踩坑指南

> [!bug] 坑一：离线推理也需要 GPU
> "离线"是指"不需要起服务"，**不是**指"不需要 GPU"。模型推理本身还是要 GPU 加速的。macOS 用户只能用 `cpu_offload_gb` 参数在 CPU 上慢慢跑。

> [!bug] 坑二：一次性传入太多 prompt 会导致 OOM
> `llm.generate()` 虽然会自动 batch，但如果传入 100 万条 prompt，在 `add_request` 阶段就会尝试把所有请求加入队列，可能把 CPU 内存撑爆。
>
> **解决方案**：分批处理，每批 5000 条。

> [!bug] 坑三：Pooling 模型必须指定 `runner="pooling"`
> 分类、嵌入、评分这些模型不是生成式模型，必须传 `runner="pooling"`，否则会报错。

> [!bug] 坑四：GGUF 量化模型的加载方式不同
> vLLM 支持 GGUF 量化模型，但加载格式是 `repo_id:quant_type`，而且需要单独指定 tokenizer。

## 选型决策：离线 vs 在线

> [!question]- 什么时候该用离线推理？
> - ✅ 数据量 > 100 条
> - ✅ 不需要流式输出
> - ✅ 任务可以等（分钟级甚至小时级）
> - ✅ 结果存文件/数据库，不直接给用户

> [!question]- 什么时候该用在线服务？
> - ❌ 用户在等回答（秒级延迟要求）
> - ❌ 需要流式输出（打字机效果）
> - ❌ 多个客户端并发调用

## 总结

> [!quote] 类比
> 在线服务是**出租车**，随叫随到但运力有限；离线推理是**货运列车**，一次拉满，吞吐惊人。
>
> 你去超市买东西，打车就行。但你要运一仓库的货，火车才是正解。
>
> **关键不在于哪个"更好"，而在于选对工具。**
