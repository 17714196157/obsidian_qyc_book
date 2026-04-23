---
title: "vLLM离线推理很好用，不要再批量调用在线服务了"
source: "https://mp.weixin.qq.com/s/acviYy8i_1hWja57_VbCnQ"
author:
  - "[[有点文艺细菌的码]]"
published:
created: 2026-04-23
description: "说实话，我以前一直搞错了 vLLM 的用法以为 vLLM 就是 vllm serve 起个服务再调 API？"
tags:
  - "clippings"
---
原创 有点文艺细菌的码 *2026年4月20日 23:54*

说实话，我以前一直搞错了 vLLM 的用法

以为 vLLM 就是 vllm serve 起个服务再调 API？那你可能错过了一半的性能。

文章末尾有彩蛋

## 先聊个痛点

你有没有过这种经历——

手头有 10 万条数据要做文本分类，或者要把一批用户评论生成摘要。你第一反应是什么？大概率是：

1.vllm serve 起一个 OpenAI 兼容的服务

2.写个 Python 脚本循环调 API

3.等着……慢慢等……

说实话，这方式能用，但有点像开卡车送快递——车是好车，但你只装了一箱货。

问题在哪？ 在线服务（Online Serving）是为实时交互设计的。它需要处理并发、管理连接、做请求排队……但当你只是要跑一批数据的时候，这些"服务"开销全是浪费。

这时候你需要的，是 vLLM 的离线推理（Offline Inference）。

## 离线推理到底是啥？

一句话说清楚：离线推理就是直接用 Python 调模型，不经过网络服务层。

看源码最直观。vLLM 的离线推理入口是 LLM 类，定义在 vllm/entrypoints/llm.py：

\# 源码第 106 行  
```
class LLM:  
"""An LLM for generating texts from given prompts and sampling parameters.  
  
Note:  
This class is intended to be used for offline inference. For online  
serving, use the AsyncLLMEngine class instead.  
"""
```

注意最后那句注释——vLLM 官方自己说了：LLM 类是给离线推理用的，在线服务请用 AsyncLLMEngine。

最简单的用法，3 行代码：

```
from vllm import LLM, SamplingParams  
  
llm = LLM(model="facebook/opt-125m")  
outputs = llm.generate(\["Hello, my name is"\], SamplingParams(temperature=0.8))

```
没有 HTTP 服务器，没有 API 调用，没有网络延迟。直接在进程内把活干完。

![[公众号文章/assets/vLLM离线推理很好用，不要再批量调用在线服务了/1e58ef1e26189443e67985c3c7a1a2de_MD5.webp]]

## 离线 vs 在线：到底差在哪？

我用一张表说清楚核心区别：

| 维度 | 离线推理（LLM 类） | 在线服务（vllm serve） |
| --- | --- | --- |
| 调用方式 | Python 函数调用 | HTTP API（OpenAI 兼容） |
| 网络开销 | 零 | 每次请求都有序列化/反序列化 |
| 批量处理 | 原生支持，自动 batch | 需要客户端自己管理并发 |
| 延迟 | 仅模型推理时间 | 模型推理 + 网络传输 + 请求排队 |
| 吞吐量 | 高（GPU 利用率拉满） | 受限于服务层调度 |
| 适用场景 | 批量数据处理 | 实时对话、API 服务 |
| 部署复杂度 | 一个 Python 脚本 | 需要管理服务进程、端口、监控 |

说真的，如果你不是在做 ChatGPT 那种实时对话产品，离线推理的性价比远高于在线服务。

## 从源码看本质差异

离线推理的核心循环在 \_run\_engine 方法里（llm.py 第 1745 行）：
```

def \_run\_engine(self, output\_type, \*, use\_tqdm=True):  
while self.llm\_engine.has\_unfinished\_requests():  
step\_outputs = self.llm\_engine.step()  
for output in step\_outputs:  
if output.finished:  
outputs.append(output)

```
就一个 while 循环，不停地调用 engine.step() 推进推理，直到所有请求完成。没有异步、没有事件循环、没有 HTTP 协议。纯同步，纯暴力，纯高效。

在线服务则用 AsyncLLMEngine（vllm/v1/engine/async\_llm.py）：

```python
class AsyncLLM(EngineClient):  
"""An asynchronous wrapper for the vLLM engine."""  
def \_\_init\_\_(self,...):  
	self.engine\_core = EngineCoreClient.make\_async\_mp\_client(...)  
	self.input\_processor = InputProcessor(...)  
	self.output\_processor = OutputProcessor(...)
```

异步引擎需要管理后台进程、处理流式输出、维护请求状态……这些都是为了让"实时交互"更流畅。但如果你不需要实时性，这些全是额外开销。

## 离线推理到底快在哪？

## 1\. 自动批量调度——你只管丢数据

这是我觉得最爽的设计。看 generate 方法的源码注释：

def generate(self, prompts, sampling\_params=None,...):  
"""This class automatically batches the given prompts, considering  
the memory constraint. For the best performance, put all of your  
prompts into a single list and pass it to this method.  
"""

"automatically batches"——自动批处理！你把 10 万条 prompt 丢进去，vLLM 内部根据 GPU 显存自动决定每次 batch 多少条，你完全不用操心。

而用在线服务呢？你得自己写并发逻辑，控制 max\_concurrency，还得处理限流和超时。一天下来光调参就能让你怀疑人生。

## 2\. 显存利用率拉满

离线推理默认 gpu\_memory\_utilization=0.9，也就是 90% 的 GPU 显存都给 KV Cache 用。在线服务因为要预留资源处理突发请求，实际利用率通常只有 60-70%。

## 3\. 零网络开销

在线服务的请求链路：客户端 → HTTP/JSON 序列化 → 网络传输 → 服务端反序列化 → 推理 → 序列化 → 网络传输 → 客户端反序列化

离线推理：Python 列表 → 推理 → Python 列表

说句大实话，对于本地批量任务，HTTP 那一圈折腾完全是白费。

## 离线推理的 6 大使用场景

## 场景一：批量文本生成

最经典的场景。你有 10 万条用户评论要生成摘要，或者 5 万个产品描述要做风格改写。

```
from vllm import LLM, SamplingParams  
  
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct")  
prompts = \[f"请为以下评论生成摘要：\\n{review}" for review in reviews\]  
outputs = llm.generate(prompts, SamplingParams(max\_tokens=200, temperature=0.3))
```

一行 llm.generate()，10 万条数据一次搞定。

## 场景二：文本分类

vLLM 的离线推理不只是文本生成，还支持分类任务：
```

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling", enforce\_eager=True)  
outputs = llm.classify(prompts)  
for prompt, output in zip(prompts, outputs):  
probs = output.outputs.probs # 直接拿到分类概率
```

注意 runner="pooling" 这个参数——分类模型不需要自回归生成，用的是 pooling runner，效率更高。

## 场景三：文本嵌入（Embedding）

RAG 系统的第一步：给文档建向量索引。
```

llm = LLM(model="intfloat/e5-small", runner="pooling", enforce\_eager=True)  
outputs = llm.embed(prompts)  
for prompt, output in zip(prompts, outputs):  
embeds = output.outputs.embedding # 拿到向量表示
```

## 场景四：Reranking / 评分

搜索排序、RAG 检索后的精排：
```

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling", enforce\_eager=True)  
outputs = llm.score(query, documents) # 输出相关性分数

```
## 场景五：Reward Model 评分

RLHF 训练中给回答打分：
```

llm = LLM(model="internlm/internlm2-1\_8b-reward", runner="pooling", enforce\_eager=True)  
outputs = llm.reward(prompts) # 输出 reward 分数

```
## 场景六：Chat 批量推理

多轮对话也能批量跑：

conversation = \[  
{"role": "system", "content": "You are a helpful assistant"},  
{"role": "user", "content": "Hello"},  
{"role": "assistant", "content": "Hello! How can I assist you today?"},  
{"role": "user", "content": "Write an essay about higher education."},  
\]  
conversations = \[conversation for \_ in range(10)\]  
outputs = llm.chat(conversations, sampling\_params, use\_tqdm=True)

llm.chat() 会自动应用模型的 chat template，你不用手动拼 prompt。

## 离线推理的原理：vLLM 是怎么做到高效的？

## 核心：PagedAttention + 连续批处理

vLLM 的核心优化是 PagedAttention（分页注意力机制）。它把 KV Cache 像操作系统管理内存页一样管理，避免了传统推理框架中 KV Cache 的显存碎片问题。

在离线推理中，这体现为 \_run\_engine 循环中的连续批处理（Continuous Batching）：

4.所有 prompt 一次性加入引擎

5.每个 step() 调用中，调度器根据当前显存状态决定哪些请求可以并行

6.已完成的请求立刻释放显存，新请求立即补上

7.循环直到所有请求完成

这意味着：你丢 10 万条 prompt 进去，vLLM 不会等你全部处理完才返回。每条完成的结果都会即时收集。

## 新特性：enqueue + wait\_for\_completion

vLLM 最近还加了异步离线推理的模式：

\# 先入队，不等待  
request\_ids = llm.enqueue(prompts, sampling\_params)  
\# 可以做别的事情……  
\# 等待完成  
outputs = llm.wait\_for\_completion()

这给离线推理增加了灵活性——你可以在推理的同时做数据预处理，不需要等全部完成才继续。

## CPU Offload：小显存跑大模型

这是个让我特别惊喜的设计：

cpu\_offload\_gb: The size (GiB) of CPU memory to use for offloading  
the model weights. This virtually increases the GPU memory space  
you can use to hold the model weights, at the cost of CPU-GPU data  
transfer for every forward pass.

翻译成人话： 你有 24GB 的 GPU，设 cpu\_offload\_gb=10，等于变出一个 34GB 的"虚拟 GPU"，能跑原来放不下的大模型。对于 13B 的模型（BF16 需要至少 26GB），这个功能简直是救命。

## 离线推理的 5 种实现方式

## 方式一：LLM 类（最简单，最推荐）

from vllm import LLM, SamplingParams  
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")  
outputs = llm.generate(prompts, SamplingParams(max\_tokens=256))

3 行代码，零配置。适合 90% 的场景。

## 方式二：LLMEngine 类（更细粒度的控制）

from vllm import LLMEngine, SamplingParams  
engine = LLMEngine.from\_engine\_args(engine\_args)  
engine.add\_request(str(request\_id), prompt, sampling\_params)  
while engine.has\_unfinished\_requests():  
request\_outputs = engine.step()

LLM 类是对 LLMEngine 的高层封装。如果你需要更精细的控制，直接用 LLMEngine。

## 方式三：数据并行（Data Parallel）

llm = LLM(model="ibm-research/PowerMoE-3b",  
data\_parallel\_size=2, tensor\_parallel\_size=2)

多 GPU 数据并行，每个 rank 处理不同分片的数据。

## 方式四：Ray Data 集群批处理

from ray.data.llm import build\_llm\_processor, vLLMEngineProcessorConfig  
config = vLLMEngineProcessorConfig(  
model\_source="unsloth/Llama-3.1-8B-Instruct",  
concurrency=1, batch\_size=64,  
)  
vllm\_processor = build\_llm\_processor(config,...)  
ds = vllm\_processor(ray\_data\_dataset)

这是生产级大规模批处理的首选方案。支持自动分片、负载均衡、容错重试。

## 方式五：分离式预填充（Disaggregated Prefill）

把"理解 prompt"和"生成回答"拆到不同 GPU 上，预填充阶段吃计算，解码阶段吃带宽，各司其职。适合超长 prompt 的场景。

## 踩坑实录

## 坑一：离线推理也需要 GPU

离线推理的"离线"是指"不需要起服务"，不是指"不需要 GPU"。模型推理本身还是要 GPU 加速的。macOS 用户只能用 cpu\_offload\_gb 参数在 CPU 上慢慢跑。

## 坑二：一次性传入太多 prompt 会导致 OOM

llm.generate() 虽然会自动 batch，但如果你传入 100 万条 prompt，在 add\_request 阶段就会尝试把所有请求加入队列，可能把 CPU 内存撑爆。

解决方案： 分批处理，每批 5000 条。

## 坑三：Pooling 模型必须指定 runner="pooling"

分类、嵌入、评分这些模型不是生成式模型，必须传 runner="pooling"，否则会报错。

## 坑四：GGUF 量化模型的加载方式不同

vLLM 支持 GGUF 量化模型，但加载格式是 repo\_id:quant\_type，而且需要单独指定 tokenizer。

## 什么时候该用离线推理？什么时候该用在线服务？

一句话：如果你的任务不需要"实时响应"，就用离线推理。

✅ 数据量 > 100 条 → 离线推理

✅ 不需要流式输出 → 离线推理

✅ 任务可以等（分钟级甚至小时级） → 离线推理

✅ 结果存文件/数据库，不直接给用户 → 离线推理

❌ 用户在等回答（秒级延迟要求） → 在线服务

❌ 需要流式输出（打字机效果） → 在线服务

❌ 多个客户端并发调用 → 在线服务

说真的，80% 的 LLM 应用场景里，离线推理就够用了。很多人起在线服务只是因为"习惯这样用"，并不是真的需要。

## 写在最后

vLLM 的离线推理让我想到一个类比——在线服务是出租车，随叫随到但运力有限；离线推理是货运列车，一次拉满，吞吐惊人。

你去超市买东西，打车就行。但你要运一仓库的货，火车才是正解。

关键不在于哪个"更好"，而在于选对工具。

vLLM 源码里那个简洁的 while 循环，在 GPU 上疯狂地调用 step()，每一次迭代都在榨干显存的最后一滴性能。没有 HTTP 的序列化开销，没有请求队列的等待时间，没有服务层的调度延迟。

这不是什么高深的优化技巧，这是最基本的工程常识——能少走一步，就少走一步。

关注我，免费领取全套的vllm推理实战手册

**微信扫一扫赞赏作者**

继续滑动看下一个

有点文艺细菌的码农

向上滑动看下一个