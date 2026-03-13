---
title: "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"
source: "https://vllm.ai/blog/anatomy-of-vllm"
author:
  - "[[vLLM Team]]"
published: 2025-09-05
tags:
  - "clippings"
---
> **Note:** Originally posted on [Aleksa Gordic's website](https://www.aleksagordic.com/blog/vllm).  
> **注意：** 最初发布在 [Aleksa Gordic 的网站](https://www.aleksagordic.com/blog/vllm) 上。

### From paged attention, continuous batching, prefix caching, specdec, etc. to multi-GPU, multi-node dynamic serving at scale从分页关注、连续批处理、前缀缓存、specdec 等，到多 GPU、多节点动态扩展的规模。

In this post, I'll gradually introduce all of the core system components and advanced features that make up a modern high-throughput LLM inference system. In particular I'll be doing a breakdown of how vLLM [\[1\]](https://vllm.ai/blog/#ref-1) works.  
在这篇文章中，我将逐步介绍构成现代高吞吐量LLM推理系统的所有核心组件和高级功能。特别是，我将分解 vLLM [\[1\]](https://vllm.ai/blog/#ref-1) 的工作原理。

This post is the first in a series. It starts broad and then layers in detail (following an inverse-pyramid approach) so you can form an accurate high-level mental model of the complete system without drowning in minutiae.  
这篇文章是该系列的第一篇。它从宏观开始，然后逐步细化（采用倒金字塔结构），以便您可以在不陷入细节的情况下形成一个对整个系统的准确高级心理模型。

Later posts will dive into specific subsystems.  
后续文章将深入探讨特定的子系统。

This post is structured into five parts:  
本文分为五个部分：

1. [LLM engine & engine core](https://vllm.ai/blog/#llm-engine--engine-core): fundamentals of vLLM (scheduling, paged attention, continuous batching, etc.)  
	[LLM 引擎 & 引擎核心](https://vllm.ai/blog/#llm-engine--engine-core) ：vLLM 的基础（调度、分页注意力、连续批处理等）
2. [Advanced features](https://vllm.ai/blog/#advanced-features--extending-the-core-engine-logic): chunked prefill, prefix caching, guided & speculative decoding, disaggregated P/D  
	[高级功能](https://vllm.ai/blog/#advanced-features--extending-the-core-engine-logic) ：分块预填充、前缀缓存、引导与推测解码、解耦 P/D
3. [Scaling up](https://vllm.ai/blog/#from-uniprocexecutor-to-multiprocexecutor): from single-GPU to multi-GPU execution  
	[扩展规模](https://vllm.ai/blog/#from-uniprocexecutor-to-multiprocexecutor) ：从单 GPU 执行到多 GPU 执行
4. [Serving layer](https://vllm.ai/blog/#distributed-system-serving-vllm): distributed / concurrent web scaffolding  
	[服务层](https://vllm.ai/blog/#distributed-system-serving-vllm) ：分布式/并发 Web 脚手架
5. [Benchmarks and auto-tuning](https://vllm.ai/blog/#benchmarks-and-auto-tuning---latency-vs-throughput): measuring latency and throughput  
	[基准测试和自动调整](https://vllm.ai/blog/#benchmarks-and-auto-tuning---latency-vs-throughput) ：测量延迟和吞吐量

> **Note:** \* Analysis is based on [commit 42172ad](https://github.com/vllm-project/vllm/tree/42172ad) (August 9th, 2025).  
> **注意：** \* 分析基于 [提交 42172ad](https://github.com/vllm-project/vllm/tree/42172ad) （2025 年 8 月 9 日）。
> 
> - Target audience: anyone curious about how state-of-the-art LLM engines work, as well as those interested in contributing to vLLM, SGLang, etc.  
> 	目标受众：任何对最先进的 LLM 引擎工作原理感兴趣的人，以及那些有兴趣为 vLLM、SGLang 等做出贡献的人。
> - I'll focus on the [V1 engine](https://docs.vllm.ai/en/latest/usage/v1_guide.html). I also explored V0 (now [deprecated](https://github.com/vllm-project/vllm/issues/18571)), which was valuable for understanding how the project evolved, and many concepts still carry over.  
> 	我将重点关注 [V1 引擎](https://docs.vllm.ai/en/latest/usage/v1_guide.html) 。我还探索了 V0（现在 [已弃用](https://github.com/vllm-project/vllm/issues/18571) ），这对理解项目的发展历程很有价值，而且许多概念仍然适用。
> - The first section on LLM Engine / Engine Core might be a bit overwhelming/dry - but the rest of the blog has plenty examples and visuals.:)  
> 	第一部分关于 LLM 引擎/引擎核心的内容可能有点令人难以理解/枯燥——但博客的其余部分有很多示例和视觉内容。:)

## LLM Engine & Engine CoreLLM 引擎 & 引擎核心

The LLM engine is the fundamental building block of vLLM. On its own, it already enables high-throughput inference - but only in an offline setting. You can't serve it to customers over the web yet.  
LLM 引擎是 vLLM 的基本构建模块。单独来看，它已经能够实现高吞吐量推理 - 但仅限于离线环境。目前还不能通过互联网向客户提供服务。

We'll use the following offline inference snippet as our running example (adapted from [basic.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/basic.py)).  
我们将以以下离线推理代码片段作为我们的示例（改编自 [basic.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/basic.py) ）。

```python
from vllm import LLM, SamplingParams
 
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
 
def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
 
    outputs = llm.generate(prompts, sampling_params)
 
if __name__ == "__main__":
    main()
```

> **Note:** Environment vars:  
> **注意：** 环境变量：
> 
> - VLLM\_USE\_V1="1" # we're using engine V1  
> 	VLLM\_USE\_V1="1" # 我们使用 V1 引擎
> - VLLM\_ENABLE\_V1\_MULTIPROCESSING="0" # we're running in a single process  
> 	VLLM\_ENABLE\_V1\_MULTIPROCESSING="0" # 我们以单进程运行

This configuration is:此配置是：

- offline (no web/distributed system scaffolding)  
	离线（无 Web/分布式系统框架）
- synchronous (all execution happens in a single blocking process)  
	同步（所有执行都在单个阻塞过程中进行）
- single-GPU (no data/model/pipeline/expert parallelism; DP/TP/PP/EP = 1)  
	单-GPU（无数据/模型/流水线/专家并行；DP/TP/PP/EP = 1）
- using standard transformer [\[2\]](https://vllm.ai/blog/#ref-2) (supporting hybrid models like Jamba requires a more complex hybrid KV-cache memory allocator)  
	使用标准变压器 [\[2\]](https://vllm.ai/blog/#ref-2) （支持混合模型如 Jamba 需要更复杂的混合 KV 缓存内存分配器）

From here, we'll gradually build up to an online, async, multi-GPU, multi-node inference system - but still serving a standard transformer.  
从这里，我们将逐步构建一个在线、异步、多-GPU、多节点的推理系统 - 但仍然提供标准变压器。

In this example we do two things, we:  
在本例中，我们做两件事，我们：

1. Instantiate an engine 实例化一个引擎
2. Call `generate` on it to sample from the given prompts  
	在它上面调用 `generate` 以从给定的提示中采样

Let's start analyzing the constructor.  
让我们从分析构造函数开始。

### LLM Engine constructor LLM 引擎构造函数

The main components of the engine are:  
引擎的主要组件包括：

- vLLM config (contains all of the knobs for configuring model, cache, parallelism, etc.)  
	vLLM 配置（包含配置模型、缓存、并行性等所有旋钮）
- processor (turns raw inputs → `EngineCoreRequests` via validation, tokenization, and processing)  
	处理器（通过验证、分词和加工将原始输入转换为 `EngineCoreRequests` ）
- engine core client (in our running example we're using `InprocClient` which is basically == `EngineCore`; we'll gradually build up to `DPLBAsyncMPClient` which allows serving at scale)  
	引擎核心客户端（在我们的运行示例中我们使用 `InprocClient` ，它基本上等于 `EngineCore` ；我们将逐步构建到 `DPLBAsyncMPClient` ，它允许大规模提供服务）
- output processor (converts raw `EngineCoreOutputs` → `RequestOutput` that the user sees)  
	输出处理器（将原始 `EngineCoreOutputs` 转换为用户看到的 `RequestOutput` ）

> **Note:** With the V0 engine being deprecated, class names and details may shift. I'll emphasize the core ideas rather than exact signatures. I'll abstract away some but not all of those details.  
> **注意：** 由于 V0 引擎已弃用，类名和细节可能会有所变动。我将强调核心思想，而不是精确的签名。我将抽象掉一些细节，但不是全部。

Engine core itself is made up of several sub components:  
引擎核心本身由几个子组件组成：

- Model Executor (drives forward passes on the model, we're currently dealing with `UniProcExecutor` which has a single `Worker` process on a single GPU). We'll gradually build up to `MultiProcExecutor` which supports multiple GPUs  
	模型执行器（驱动模型的前向传递，我们目前处理的是 `UniProcExecutor` ，它在一个单 GPU 上有一个单独的 `Worker` 进程）。我们将逐步过渡到支持多个 GPU 的 `MultiProcExecutor`
- Structured Output Manager (used for guided decoding - we'll cover this later)  
	结构化输出管理器（用于引导解码 - 我们稍后将会介绍）
- Scheduler (decides which requests go into the next engine step) - it further contains:  
	调度器（决定哪些请求进入下一个引擎步骤） - 它还包含：
	1. policy setting - it can be either **FCFS** (first come first served) or **priority** (higher priority requests are served first)  
		策略设置 - 它可以是 **FCFS** （先到先得）或 **优先级** （优先级较高的请求先被服务）
	2. `waiting` and `running` queues  
		等待队列 `waiting` 和运行队列 `running`
	3. KV cache manager - the heart of paged attention [\[3\]](https://vllm.ai/blog/#ref-3)  
		KV 缓存管理器 - 分页注意力机制的核心 [\[3\]](https://vllm.ai/blog/#ref-3)

The KV-cache manager maintains a `free_block_queue` - a pool of available KV-cache blocks (often on the order of hundreds of thousands, depending on VRAM size and block size). During paged attention, the blocks serve as the indexing structure that map tokens to their computed KV cache blocks.  
KV-cache 管理器维护一个 `空闲块队列` - 一组可用的 KV-cache 块（通常在数十万左右，取决于 VRAM 大小和块大小）。在分页注意力机制期间，这些块作为索引结构，将标记映射到其计算出的 KV-cache 块。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/fe49ce06f12fd4586311afe0911fa52a_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

> **Note:** Block size for a standard transformer layer (non-MLA [\[4\]](https://vllm.ai/blog/#ref-4)) is computed as follows: 2 (key/value) \* `block_size` (default=16) \* `num_kv_heads` \* `head_size` \* `dtype_num_bytes` (e.g. 2 for bf16)  
> **注意：** 标准 Transformer 层的块大小（非 MLA [\[4\]](https://vllm.ai/blog/#ref-4) ）计算如下：2（键/值）\* `block_size` （默认=16）\* `num_kv_heads` \* `head_size` \* `dtype_num_bytes` （例如，bf16 为 2）

During model executor construction, a `Worker` object is created, and three key procedures are executed. (Later, with `MultiProcExecutor`, these same procedures run independently on each worker process across different GPUs.)  
在模型执行器构建过程中，会创建一个 `Worker` 对象，并执行三个关键步骤。（稍后，使用 `MultiProcExecutor` ，这些相同的步骤将在每个工作进程上独立运行，这些工作进程分布在不同的 GPU 上。）

1. Init device:初始化设备：
- Assign a CUDA device (e.g. "cuda:0") to the worker and check that the model dtype is supported (e.g. bf16)  
	分配一个 CUDA 设备（例如 "cuda:0"）给工作进程，并检查模型数据类型是否受支持（例如 bf16）
- Verify enough VRAM is available, given the requested `gpu_memory_utilization` (e.g. 0.8 → 80% of total VRAM)  
	验证是否足够的 VRAM 可用，给定请求的 `gpu_memory_utilization` （例如 0.8 → 总 VRAM 的 80%）
- Set up distributed settings (DP / TP / PP / EP, etc.)  
	设置分布式设置（DP / TP / PP / EP 等）
- Instantiate a `model_runner` (holds the sampler, KV cache, and forward-pass buffers such as `input_ids`, `positions`, etc.)  
	实例化一个 `model_runner` （包含采样器、KV 缓存和前向传递缓冲区，如 `input_ids` 、 `positions` 等）
- Instantiate an `InputBatch` object (holds CPU-side forward-pass buffers, block tables for KV-cache indexing, sampling metadata, etc.)  
	实例化一个 `InputBatch` 对象（包含 CPU 端前向传递缓冲区、KV 缓存索引的块表、采样元数据等）
1. Load model:加载模型：
- Instantiate the model architecture  
	实例化模型架构
- Load the model weights 加载模型权重
- Call model.eval() (PyTorch's inference mode)  
	调用模型.eval()（PyTorch 的推理模式）
- Optional: call torch.compile() on the model  
	可选：在模型上调用 torch.compile()
1. Initialize KV cache 初始化 KV 缓存
- Get per-layer KV-cache spec. Historically this was always `FullAttentionSpec` (homogeneous transformer), but with hybrid models (sliding window, Transformer/SSM like Jamba) it became more complex (see Jenga [\[5\]](https://vllm.ai/blog/#ref-5))  
	获取每层的 KV 缓存规范。历史上这始终是 `FullAttentionSpec` （同质化 Transformer），但随着混合模型（滑动窗口、类似 Jamba 的 Transformer/SSM）的出现，它变得更加复杂（参见 Jenga [\[5\]](https://vllm.ai/blog/#ref-5) ）
- Run a dummy/profiling forward pass and take a GPU memory snapshot to compute how many KV cache blocks fit in available VRAM  
	运行一个模拟/性能分析的前向传递，并获取 GPU 内存快照以计算多少 KV 缓存块可以放入可用的 VRAM
- Allocate, reshape and bind KV cache tensors to attention layers  
	分配、重塑并绑定 KV 缓存张量到注意力层
- Prepare attention metadata (e.g. set the backend to FlashAttention) later consumed by kernels during the fwd pass  
	准备注意力元数据（例如，将后端设置为 FlashAttention），稍后由内核在正向传递期间使用
- Unless `--enforce-eager` is provided, for each of warmup batch sizes do a dummy run and capture CUDA graphs. CUDA graphs record the whole sequence of GPU work into a DAG. Later during fwd pass we launch/replay pre-baked graphs and cut on kernel launch overhead and thus improve latency.  
	除非提供 `--enforce-eager` ，否则对于每个预热批次大小，都要进行一次模拟运行并捕获 CUDA 图。CUDA 图记录整个 GPU 工作序列到一个 DAG 中。在后续的前向传递过程中，我们启动/回放预制的图，减少内核启动开销，从而提高延迟。

I've abstracted away many low-level details here — but these are the core pieces I'll introduce now, since I'll reference them repeatedly in the following sections.  
我已经抽象掉了许多低级细节——但这些都是我现在要介绍的核心理念，因为接下来几节中我会反复引用它们。

Now that we have the engine initialized let's proceed to the `generate` function.  
现在我们已经初始化了引擎，让我们继续到 `生成` 函数。

### Generate function 生成函数

The first step is to validate and feed requests into the engine. For each prompt we:  
第一步是验证并将请求输入到引擎中。对于每个提示，我们：

1. Create a unique request ID and capture its arrival time  
	创建一个唯一的请求 ID 并捕获其到达时间
2. Call an input preprocessor that tokenizes the prompt and returns a dictionary containing `prompt`, `prompt_token_ids`, and a `type` (text, tokens, embeds, etc.)  
	调用一个输入预处理程序，该程序将提示进行分词并返回一个包含 `prompt` 、 `prompt_token_ids` 和 `type` （文本、令牌、嵌入等）的字典
3. Pack this info into an `EngineCoreRequest`, adding priority, sampling params, and other metadata  
	将此信息打包到 `EngineCoreRequest` 中，添加优先级、采样参数和其他元数据
4. Pass the request into the engine core, which wraps it in a `Request` object and sets its status to `WAITING`. This request is then added to the scheduler's `waiting` queue (append if FCFS, or heap-push if priority)  
	将请求传递给引擎核心，它将其包装在 `Request` 对象中，并将其状态设置为 `等待` 。然后，此请求被添加到调度器的 `等待` 队列中（如果采用先来先服务，则追加；如果采用优先级，则堆推）

At this point the engine has been fed and execution can begin. In the synchronous engine example, these initial prompts are the only ones we'll process — there's no mechanism to inject new requests mid-run. In contrast, the asynchronous engine supports this (aka **continuous batching** [\[6\]](https://vllm.ai/blog/#ref-6)): after each step, both new and old requests are considered.  
到此为止，引擎已经准备好，可以开始执行。在同步引擎示例中，这些初始提示是我们将处理的唯一提示——在运行过程中没有机制可以注入新的请求。相比之下，异步引擎支持这一点（即 **连续批处理** [\[6\]](https://vllm.ai/blog/#ref-6) ）：在每一步之后，都会考虑新的和旧的请求。

> **Note:** Because the forward pass flattens the batch into a single sequence and custom kernels handle it efficiently, continuous batching is fundamentally supported even in the synchronous engine.  
> <强烈 id=0>注意：由于正向传播将批次展平为单个序列，并且自定义内核可以高效地处理它，即使在同步引擎中，连续批处理也得到了根本支持。

Next, as long as there are requests to process, the engine repeatedly calls its `step()` function. Each step has three stages:  
接下来，只要有处理请求，引擎就会反复调用其 `step()` 函数。每个步骤有三个阶段：

1. Schedule: select which requests to run in this step (decode, and/or (chunked) prefill)  
	调度：选择在此步骤中运行哪些请求（解码和/或（分块）预填充）
2. Forward pass: run the model and sample tokens  
	前向传播：运行模型并采样标记
3. Postprocess: append sampled token IDs to each `Request`, detokenize, and check stop conditions. If a request is finished, clean up (e.g. return its KV-cache blocks to `free_block_queue`) and return the output early  
	后处理：将采样标记 ID 附加到每个 `请求` ，反标记化，并检查停止条件。如果请求完成，清理（例如，将其 KV 缓存块返回到 `空闲块队列` ）并提前返回输出

> **Note:** Stop conditions are:  
> **注意：** 停止条件是：
> 
> - The request exceeds its length limit (`max_model_length` or its own `max_tokens`)  
> 	请求超过了其长度限制（ `max_model_length` 或其自身的 `max_tokens` ）
> - The sampled token is the EOS ID (unless `ignore_eos` is enabled -> useful for benchmarking when we want to force a generation of a certain number of out tokens)  
> 	采样令牌是 EOS ID（除非启用了 `ignore_eos` -> 当我们想强制生成一定数量的输出令牌时，这在基准测试中很有用）
> - The sampled token matches any of the `stop_token_ids` specified in the sampling parameters  
> 	采样令牌与采样参数中指定的任何 `stop_token_ids` 匹配
> - Stop strings are present in the output - we truncate the output until the first stop string appearance and abort the request in the engine (note that `stop_token_ids` will be present in the output but stop strings will not).  
> 	输出中存在停止字符串 - 我们截断输出直到第一个停止字符串出现，并在引擎中终止请求（注意， `stop_token_ids` 将出现在输出中，但停止字符串不会）。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/92aa33219b8cd3e54b5a1e60ba5d082a_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

> **Note:** In streaming mode, we would send intermediate tokens as they are generated, but we'll ignore that for now.  
> **注意：** 在流式模式下，我们会发送生成的中间标记，但现在我们将忽略这一点。

Next, we'll examine scheduling in more detail.  
接下来，我们将更详细地探讨调度。

### Scheduler 调度器

There are two main types of workloads an inference engine handles:  
推理引擎主要处理两种类型的工作负载：

1. **Prefill** requests — a forward pass over all prompt tokens. These are usually **compute-bound** (threshold depends on hardware and prompt length). At the end, we sample a single token from the probability distribution of the final token's position.  
	**预填充** 请求 — 对所有提示标记进行正向传递。这些通常 **计算密集型** （阈值取决于硬件和提示长度）。最后，我们从最终标记位置的概率分布中采样单个标记。
2. **Decode** requests — a forward pass over just the most recent token. All earlier KV vectors are already cached. These are **memory-bandwidth-bound**, since we still need to load all LLM weights (and KV caches) just to compute one token.  
	**解码** 请求——仅对最新令牌进行正向传递。所有早期的 KV 向量都已缓存。这些是 **内存带宽限制的** ，因为我们仍然需要加载所有LLM权重（和 KV 缓存）来计算一个令牌。

> **Note:** In the [benchmarking section](https://vllm.ai/blog/#benchmarks-and-auto-tuning---latency-vs-throughput) we'll analyze the so-called roofline model of GPU perf. That will go into more detail behind prefill/decode perf profiles.  
> **注意：** 在 [基准测试部分](https://vllm.ai/blog/#benchmarks-and-auto-tuning---latency-vs-throughput) ，我们将分析所谓的 GPU 性能屋顶线模型。这将更详细地介绍 prefill/decode 性能配置文件背后的细节。

The V1 scheduler can mix both types of requests in the same step, thanks to smarter design choices. In contrast, the V0 engine could only process either prefill or decode at once.  
由于更智能的设计选择，V1 调度器可以在同一步骤中混合这两种类型的请求。相比之下，V0 引擎一次只能处理 prefill 或 decode 中的一种。

The scheduler prioritizes decode requests — i.e. those already in the `running` queue. For each such request it:  
调度器优先处理解码请求——即那些已经在 `运行` 队列中的请求。对于每个此类请求，它：

1. Computes the number of new tokens to generate (not always 1, due to speculative decoding and async scheduling — more on that later).  
	计算生成新标记的数量（不一定是 1，因为存在推测性解码和异步调度——稍后详述）。
2. Calls the KV-cache manager's `allocate_slots` function (details below).  
	调用 KV 缓存管理器的 `allocate_slots` 函数（详情见下文）。
3. Updates the token budget by subtracting the number of tokens from step 1.  
	通过减去步骤 1 中的标记数量来更新标记预算。

After that, it processes prefill requests from the `waiting` queue, it:  
之后，它处理来自 `等待` 队列的预填充请求，然后：

1. Retrieves the number of computed blocks (returns 0 if prefix caching is disabled — we'll cover that later).  
	获取已计算块的数量（如果禁用前缀缓存，则返回 0 — 我们稍后会讨论这一点）。
2. Calls the KV-cache manager's `allocate_slots` function.  
	调用 KV 缓存管理器的 `allocate_slots` 函数。
3. Pops the request from waiting and moves it to running, setting its status to `RUNNING`.  
	从等待队列中弹出请求并将其移动到运行状态，将其状态设置为 `运行` 。
4. Updates the token budget.  
	更新令牌预算。

Let's now look at what `allocate_slots` does, it:  
现在让我们看看 `allocate_slots` 做了什么，它：

1. **Computes number of blocks** — determines how many new KV-cache blocks (`n`) must be allocated. Each block stores 16 tokens by default. For example, if a prefill request has 17 new tokens, we need `ceil(17/16) = 2` blocks.  
	**计算块数量** — 确定需要分配多少新的 KV 缓存块 (`n`)。每个块默认存储 16 个标记。例如，如果预填充请求有 17 个新标记，我们需要 `ceil(17/16) = 2` 个块。
2. **Checks availability** — if there aren't enough blocks in the manager's pool, exit early. Depending on whether it's a decode or prefill request, the engine may attempt recompute preemption (swap preemption was supported in V0) by evicting low-priority requests (calling `kv_cache_manager.free` which returns KV blocks to block pool), or it might skip scheduling and continue execution.  
	**检查可用性** — 如果管理器池中没有足够的块，则提前退出。根据是解码请求还是预填充请求，引擎可能会尝试重新计算抢占（V0 中支持交换抢占），通过驱逐低优先级请求（调用 `kv_cache_manager.free` 将 KV 块返回到块池），或者它可能跳过调度并继续执行。
3. **Allocates blocks** — via the KV-cache manager's coordinator, fetches the first `n` blocks from the block pool (the `free_block_queue` doubly linked list mentioned earlier). Stores to `req_to_blocks`, the dictionary mapping each `request_id` to its list of KV-cache blocks.  
	**分配块** — 通过 KV 缓存管理器的协调器，从块池（前面提到的 `free_block_queue` 双链表）中获取前 `n` 个块。存储到 `req_to_blocks` ，该字典将每个 `request_id` 映射到其 KV 缓存块列表。
![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/e1867a6666dc741014ad607b8b288019_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

We're finally ready to do a forward pass!  
我们终于准备好进行前向传播了！

### Run forward pass 前向运行

We call model executor's `execute_model`, which delegates to the `Worker`, which in turn delegates to the model runner.  
我们调用模型执行器的\`execute\_model\`，它委托给\`Worker\`，然后\`Worker\`再委托给模型运行器。

Here are the main steps:  
以下是主要步骤：

1. **Update states** — prune finished requests from `input_batch`; update misc fwd pass related metadata (e.g., KV cache blocks per request that will be used to index into paged KV cache memory).  
	**更新状态** — 从 `input_batch` 中修剪完成请求；更新与正向传播相关的元数据（例如，每个请求将用于索引分页 KV 缓存内存的 KV 缓存块）。
2. **Prepare inputs** — copy buffers from CPU→GPU; compute positions; build `slot_mapping` (more on that in example); construct attention metadata.  
	**准备输入** — 将缓冲区从 CPU 复制到 GPU；计算位置；构建 `slot_mapping` （更多内容将在示例中说明）；构造注意力元数据。
3. **Forward pass** — run the model with custom paged attn kernels. All sequences are flattened and concatenated into one long "super sequence". Position indices and attention masks ensure each sequence only attends to its own tokens, which enables continuous batching without right-padding.  
	**正向传播** — 使用自定义分页注意力内核运行模型。所有序列都被展平并连接成一个长的“超级序列”。位置索引和注意力掩码确保每个序列只关注其自身的标记，从而实现无需右填充的连续批处理。
4. **Gather last-token states** — extract hidden states for each sequence's final position and compute logits.  
	**收集最后标记状态** — 提取每个序列最后位置的隐藏状态并计算 logits。
5. **Sample** — sample tokens from computed logits as dictated by the sampling config (greedy, temperature, top-p, top-k, etc.).  
	**采样** — 根据采样配置（贪婪、温度、top-p、top-k 等）从计算的 logits 中采样标记。

Forward-pass step itself has two execution modes:  
前向传播步骤本身有两种执行模式：

1. **Eager mode** — run the standard PyTorch forward pass when eager execution is enabled.  
	**急切模式** — 当启用急切执行时运行标准的 PyTorch 前向传递。
2. **"Captured" mode** — execute/replay a pre-captured CUDA Graph when eager is not enforced (remember we captured these during engine construction in the initialize KV cache procedure).  
	**"捕获"模式** — 当不强制执行急切时执行/回放预先捕获的 CUDA 图（记住我们在初始化 KV 缓存过程中捕获了这些）。

Here is a concrete example that should make continuous batching and paged attention clear:  
这里有一个具体的例子，应该可以使连续批处理和分页注意力变得清晰：

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/b910612a59621147d9050253b17ad681_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

## Advanced Features — extending the core engine logic高级功能 — 扩展核心引擎逻辑

With the basic engine flow in place, we can now look at the advanced features.  
基本引擎流程已经就绪，现在我们可以看看高级功能。

We've already discussed preemption, paged attention, and continuous batching.  
我们已经讨论了抢占、分页注意力和连续批处理。

Next, we'll dive into:接下来，我们将深入了解：

1. Chunked prefill 分块预填充
2. Prefix caching 前缀缓存
3. Guided decoding (through grammar-constrained finite-state machines)  
	通过语法约束的有限状态机引导解码
4. Speculative decoding 投机解码
5. Disaggregated P/D (prefill/decoding)  
	分解式 P/D（预填充/解码）

### Chunked prefill 分块预填充

Chunked prefill is a technique for handling long prompts by splitting their prefill step into smaller chunks. Without it, we could end up with a single very long request monopolizing one engine step disallowing other prefill requests to run. That would postpone all other requests and increase their latency.  
分块预填充是一种处理长提示的技术，通过将预填充步骤拆分为更小的块。如果没有它，我们可能会得到一个非常长的请求，垄断一个引擎步骤，不允许其他预填充请求运行。这将推迟所有其他请求并增加它们的延迟。

For example, let each chunk contain `n` (=8) tokens, labeled with lowercase letters separated by "-". A long prompt `P` could look like `x-y-z`, where `z` is an incomplete chunk (e.g. 2 toks). Executing the full prefill for `P` would then take ≥ 3 engine steps (> can happen if it's not scheduled for execution in one of the steps), and only in the last chunked prefill step would we sample one new token.  
例如，让每个块包含 `n` （=8）个令牌，用小写字母分隔。一个长提示 `P` 可能看起来像 `x-y-z` ，其中 `z` 是一个不完整的块（例如，2 个令牌）。然后对 `P` 执行完整的预填充将需要 ≥ 3 个引擎步骤（> 可能发生，如果它没有被安排在其中一个步骤中执行），并且只有在最后一个分块预填充步骤中，我们才会采样一个新令牌。

Here is that same example visually:  
这里就是那个相同示例的视觉呈现：

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/220a6d75512e925b0668fec9cbe82522_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

Implementation is straightforward: cap the number of new tokens per step. If the requested number exceeds `long_prefill_token_threshold`, reset it to exactly that value. The underlying indexing logic (described earlier) takes care of the rest.  
实现很简单：限制每步生成的新标记数量。如果请求的数量超过 `long_prefill_token_threshold` ，则将其重置为该确切值。底层索引逻辑（前面已描述）负责处理其余部分。

In vLLM V1, you enable chunked prefill by setting `long_prefill_token_threshold` to a positive integer. (Technically, it can happen irrespective of this, if the prompt length exceeds the token budget we truncate it and run a chunked prefill.)  
在 vLLM V1 中，您可以通过将 `long_prefill_token_threshold` 设置为正整数来启用分块预填充。（技术上，无论是否设置此值，如果提示长度超过令牌预算，我们都会截断并运行分块预填充。）

### Prefix Caching 前缀缓存

To explain how prefix caching works, let's take the original code example and tweak it a bit:  
为了解释前缀缓存的工作原理，让我们稍微修改一下原始代码示例：

```python
from vllm import LLM, SamplingParams
 
long_prefix = "<a piece of text that is encoded into more than block_size tokens>"
 
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
 
def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
 
    outputs = llm.generate(long_prefix + prompts[0], sampling_params)
    outputs = llm.generate(long_prefix + prompts[1], sampling_params)
 
if __name__ == "__main__":
    main()
```

Prefix caching avoids recomputing tokens that multiple prompts share at the beginning - hence **prefix**.  
前缀缓存避免了多个提示在开头共享的标记的重新计算 - 因此 **前缀** 。

The crucial piece is the `long_prefix`: it's defined as any prefix longer than a KV-cache block (16 tokens by default). To simplify our example let's say `long_prefix` has exactly length `n x block_size` (where `n ≥ 1`).  
关键部分是 `长前缀` ：它被定义为任何比 KV 缓存块（默认为 16 个标记）更长的前缀。为了简化我们的示例，让我们假设 `长前缀` 的长度正好是 `n x block_size` （其中 `n ≥ 1` ）。

> **Note:** i.e. it perfectly aligns with block boundary - otherwise we'd have to recompute `long_prefix_len % block_size` tokens as we can't cache incomplete blocks.  
> **注意：** 即它完美地与块边界对齐——否则我们就需要重新计算 `long_prefix_len % block_size` 个标记，因为我们无法缓存不完整的块。

Without prefix caching, each time we process a new request with the same `long_prefix`, we'd recompute all `n x block_size` tokens.  
没有前缀缓存的情况下，每次我们处理一个具有相同 `长前缀` 的新请求时，我们都需要重新计算所有 `n x block_size` 个标记。

With prefix caching, those tokens are computed once (their KVs stored in KV cache paged memory) and then reused, so only the new prompt tokens need processing. This speeds up prefill requests (though it doesn't help with decode).  
使用前缀缓存，这些标记只计算一次（它们的键值对存储在键值缓存分页内存中）然后重复使用，因此只需处理新的提示标记。这加快了预填充请求（尽管它对解码没有帮助）。

How does this work in vLLM?  
这在 vLLM 中是如何工作的？

During the first `generate` call, in the scheduling stage, inside `kv_cache_manager.get_computed_blocks`, the engine invokes `hash_request_tokens`:  
在第一次 `generate` 调用期间，在调度阶段，在 `kv_cache_manager.get_computed_blocks` 内部，引擎调用 `hash_request_tokens` ：

1. This function splits the `long_prefix + prompts[0]` into 16-token chunks.  
	此函数将 `long_prefix + prompts[0]` 分割成 16 个令牌的块。
2. For each complete chunk, it computes a hash (using either the built-in hash or SHA-256, which is slower but has fewer collisions). The hash combines the previous block's hash, the current tokens, and optional metadata.  
	对于每个完整的块，它计算一个哈希（使用内置哈希或 SHA-256，SHA-256 较慢但碰撞更少）。哈希结合了上一个块的哈希、当前令牌和可选元数据。

> **Note:** optional metadata includes: MM hash, LoRA ID, cache salt (injected into hash of the first block ensures only requests with this cache salt can reuse blocks).  
> **注意：** 可选元数据包括：MM 哈希、LoRA ID、缓存盐（注入到第一个块的哈希中，确保只有带有此缓存盐的请求可以重用块）。

1. Each result is stored as a `BlockHash` object containing both the hash and its token IDs. We return a list of block hashes.  
	每个结果都存储为一个包含哈希和其令牌 ID 的 `BlockHash` 对象。我们返回一个区块哈希列表。

The list is stored in `self.req_to_block_hashes[request_id]`.  
该列表存储在 `self.req_to_block_hashes[request_id]` 。

Next, the engine calls `find_longest_cache_hit` to check if any of these hashes already exist in `cached_block_hash_to_block`. On the first request, no hits are found.  
接着，引擎调用 `find_longest_cache_hit` 来检查这些哈希是否已存在于 `cached_block_hash_to_block` 中。在第一次请求时，没有找到匹配项。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/ec988ef4c750542e3ce1ab2fd257ffe0_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

Then we call `allocate_slots` which calls `coordinator.cache_blocks`, which associates the new `BlockHash` entries with allocated KV blocks and records them in `cached_block_hash_to_block`.  
然后我们调用 `allocate_slots` ，它调用 `coordinator.cache_blocks` ，将新的 `BlockHash` 条目与分配的 KV 块关联，并在 `cached_block_hash_to_block` 中记录它们。

Afterwards, the forward pass will populate KVs in paged KV cache memory corresponding to KV cache blocks that we allocated above.  
之后，前向传播将填充我们上面分配的 KV 缓存块对应的分页 KV 缓存内存中的 KVs。

> **Note:** After many engine steps it'll allocate more KV cache blocks but it doesn't matter for our example because the prefix has diverged immediately after `long_prefix`.  
> 注意：经过许多引擎步骤后，它会分配更多的 KV 缓存块，但在我们的例子中并不重要，因为前缀在 `long_prefix` 之后立即发生了分歧。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/5556fe3d7b029e9e02ff0c55936d40f9_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

On a second `generate` call with the same prefix, steps 1-3 repeat, but now `find_longest_cache_hit` finds matches for all `n` blocks (via linear search). The engine can reuse those KV blocks directly.  
在第二个具有相同前缀的 `generate` 调用中，步骤 1-3 重复，但现在 `find_longest_cache_hit` 为所有 `n` 块查找匹配项（通过线性搜索）。引擎可以直接重用这些 KV 块。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/835186b62496f5a8c7212dc25979b7fd_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

If the original request were still alive, the reference count for those blocks would increment (e.g. to 2). In this example, the first request has already completed, so the blocks were freed back to the pool and their reference counts set back to 0. Because we were able to retrieve them from `cached_block_hash_to_block` we know they're valid (the logic of the KV cache manager is setup in such a way), so we just remove them from `free_block_queue` again.  
如果原始请求仍然存在，这些块的引用计数将增加（例如，增加到 2）。在这个例子中，第一个请求已经完成，因此这些块被释放回池中，它们的引用计数被重置为 0。因为我们能够从 `cached_block_hash_to_block` 中检索到它们，我们知道它们是有效的（KV 缓存管理器的逻辑就是这样设置的），所以我们再次将它们从 `free_block_queue` 中移除。

> \[!NOTE\] Advanced note: KV-cache blocks become invalid only when they're about to be reallocated from the `free_block_queue` (which pops from the left) and we discover the block still has an associated hash and is present in `cached_block_hash_to_block`. At that moment, we clear the block's hash and remove its entry from `cached_block_hash_to_block`, ensuring it can't be reused via prefix caching (at least not for that old prefix).  
> \[!注意\] 高级笔记：KV 缓存块仅在即将从 `free_block_queue` （从左侧弹出）重新分配时失效，并且我们发现该块仍然关联着哈希并且存在于 `cached_block_hash_to_block` 中。此时，我们清除块的哈希并从 `cached_block_hash_to_block` 中移除其条目，确保它不能通过前缀缓存（至少不是针对那个旧前缀）被重复使用。

And that's the gist of prefix caching: don't recompute prefixes you've already seen — just reuse their KV cache!  
这就是前缀缓存的精髓：不要重新计算你已经见过的前缀——只需重用它们的 KV 缓存！

If you understood this example you also understood how paged attention works.  
如果你理解了这个例子，那么你也理解了分页注意力机制的工作原理。

Prefix caching is enabled by default. To disable it: `enable_prefix_caching = False`.  
默认启用前缀缓存。要禁用它： `enable_prefix_caching = False` 。

### Guided Decoding (FSM) 指导解码（有限状态机）

Guided decoding is a technique where, at each decoding step, the logits are constrained by a grammar-based finite state machine. This ensures that only tokens allowed by the grammar can be sampled.  
指导解码是一种技术，在每个解码步骤中，通过基于语法的有限状态机对 logits 进行约束。这确保只能采样语法允许的标记。

It's a powerful setup: you can enforce anything from regular grammars (Chomsky type-3, e.g. arbitrary regex patterns) all the way up to context-free grammars (type-2, which cover most programming languages).  
这是一个强大的设置：您可以强制执行从常规语法（乔姆斯基类型-3，例如任意正则表达式模式）到上下文无关语法（类型-2，涵盖大多数编程语言）的一切。

To make this less abstract, let's start with the simplest possible example, building on our earlier code:  
为了使这个问题更具体，让我们从最简单的例子开始，基于我们之前的代码：

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
 
prompts = [
    "This sucks",
    "The weather is beautiful",
]
 
guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
 
def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
 
    outputs = llm.generate(prompts, sampling_params)
 
if __name__ == "__main__":
    main()
```

In the toy example I gave (assume character-level tokenization): at prefill, the FSM masks logits so only "P" or "N" are viable. If "P" is sampled, the FSM moves to the "Positive" branch; next step only "o" is allowed, and so on.  
在我在玩具示例中给出的例子（假设字符级别的分词）：在预填充时，FSM 会屏蔽 logits，因此只有“P”或“N”是可行的。如果采样到“P”，FSM 会移动到“Positive”分支；下一步只允许“o”，以此类推。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/ecaeba4122c2e79c10924cef44dae12b_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

How this works in vLLM:  
vLLM 中它是如何工作的：

1. At LLM engine construction, a `StructuredOutputManager` is created; it has access to the tokenizer and maintains a `_grammar_bitmask` tensor.  
	在 LLM 引擎构建时，创建了一个 `结构化输出管理器` ；它有权访问分词器并维护一个 `_grammar_bitmask` 张量。
2. When adding a request, its status is set to `WAITING_FOR_FSM` and `grammar_init` selects the backend compiler (e.g., `xgrammar` [\[7\]](https://vllm.ai/blog/#ref-7); note that backends are 3rd party code).  
	当添加请求时，其状态设置为 `等待 FSM` 和 `语法初始化` 选择后端编译器（例如， `xgrammar` [\[7\]](https://vllm.ai/blog/#ref-7) ；请注意后端是第三方代码）。
3. The grammar for this request is compiled asynchronously.  
	此请求的语法是异步编译的。
4. During scheduling, if the async compile has completed, the status switches to `WAITING` and `request_id` is added to `structured_output_request_ids`; otherwise it's placed in `skipped_waiting_requests` to retry on next engine step.  
	在调度过程中，如果异步编译已完成，状态将切换到 `等待` ，并将 `request_id` 添加到 `structured_output_request_ids` 中；否则，它将被放置在 `skipped_waiting_requests` 中，以便在下一次引擎步骤中重试。
5. After the scheduling loop (still inside scheduling), if there are FSM requests, the `StructuredOutputManager` asks the backend to prepare/update `_grammar_bitmask`.  
	在调度循环之后（仍在调度中），如果有 FSM 请求， `StructuredOutputManager` 将要求后端准备/更新 `_grammar_bitmask` 。
6. After the forward pass produces logits, xgr\_torch\_compile's function expands the bitmask to vocab size (32x expansion ratio because we use 32 bit integers) and masks disallowed logits to –∞.  
	在前向传播生成 logits 之后，xgr\_torch\_compile 函数将位掩码扩展到词汇表大小（因为我们使用 32 位整数，所以扩展比为 32 倍）并将不允许的 logits 掩码设置为-∞。
7. After sampling the next token, the request's FSM is advanced via `accept_tokens`. Visually we move to the next state on the FSM diagram.  
	在采样下一个标记后，请求的 FSM 通过 `accept_tokens` 进行状态转换。在 FSM 图上，我们移动到下一个状态。

Step 6 deserves further clarification.  
第 6 步需要进一步说明。

If `vocab_size = 32`, `_grammar_bitmask` is a single integer; its binary representation encodes which tokens are allowed ("1") vs disallowed ("0"). For example, "101…001" expands to a length-32 array `[1, 0, 1, ..., 0, 0, 1]`; positions with 0 get logits set to –∞. For larger vocabularies, multiple 32-bit words are used and expanded/concatenated accordingly. The backend (e.g., `xgrammar`) is responsible for producing these bit patterns using the current FSM state.  
如果 `vocab_size = 32` ， `_grammar_bitmask` 是一个单独的整数；其二进制表示编码了哪些标记是允许的（"1"）与不允许的（"0"）。例如，"101…001" 展开为长度为 32 的数组 `[1, 0, 1, ..., 0, 0, 1]` ；位置为 0 的 logits 被设置为-∞。对于更大的词汇表，使用多个 32 位单词，并相应地进行展开/连接。后端（例如， `xgrammar` ）负责使用当前的 FSM 状态生成这些位模式。

> **Note:** Most of the complexity here is hidden in the 3rd party libs like xgrammar.  
> **注意：** 这里的大部分复杂性都隐藏在第三方库 xgrammar 中。

Here is an even simpler example with vocab\_size = 8 and 8-bit integers (for those of you who like my visuals):  
这里有一个更简单的例子，vocab\_size = 8，以及 8 位整数（对于那些喜欢我的视觉效果的人来说）：

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/713302beaa901ac1fad1dd02ca1ee312_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

You can enable this in vLLM by passing in a desired `guided_decoding` config.  
您可以在 vLLM 中通过传递所需的 `guided_decoding` 配置来启用此功能。

### Speculative Decoding 推测性解码

In autoregressive generation, each new token requires a forward pass of the large LM. This is expensive — every step reloads and applies all model weights just to compute a single token! (assuming batch size == 1, in general it's `B`)  
在自回归生成中，每个新标记都需要对大型语言模型进行一次正向传递。这很昂贵——每一步都要重新加载并应用所有模型权重，只是为了计算一个标记！（假设批处理大小为 1，通常它小于 `B` ）

Speculative decoding [\[8\]](https://vllm.ai/blog/#ref-8) speeds this up by introducing a smaller draft LM. The draft proposes `k` tokens cheaply. But we don't ultimately want to sample from the smaller model — it's only there to guess candidate continuations. The large model still decides what's valid.  
推测性解码 [\[8\]通过引入一个较小的草稿语言模型来加快这个过程。草稿廉价地提出 `k` 个标记。但我们最终不希望从较小的模型中采样——它只是为了猜测候选续集。大型模型仍然决定什么有效。](https://vllm.ai/blog/#ref-8)

Here are the steps:下面是步骤：

1. **Draft**: run the small model on the current context and propose `k` tokens  
	**草案** ：在当前上下文中运行小模型并提议 `k` 个标记
2. **Verify**: run the large model once on context + `k` draft tokens. This produces probabilities for those `k` positions plus one extra (so we get `k+1` candidates)  
	**验证** ：在上下文加上 `k` 个草案标记上运行大模型一次。这为这些 `k` 个位置以及一个额外的位置产生概率（因此我们得到 `k+1` 个候选）
3. **Accept/reject**: going from left to right over the `k` draft tokens:  
	**接受/拒绝** ：从左到右遍历 `k` 个草案标记：
- If the large model's probability for the draft token ≥ the draft's probability, accept it  
	如果大模型的草稿标记概率 ≥ 草稿的概率，则接受它
- Otherwise, accept it with probability `p_large(token)/p_draft(token)`  
	否则，以概率 `p_large(token)/p_draft(token)` 接受它
- Stop at the first rejection, or accept all `k` draft tokens  
	在第一次拒绝时停止，或接受所有 `k` 个草稿标记
- If all `k` draft tokens are accepted, also sample the extra `(k+1)` -th token "for free" from the large model (we already computed that distribution)  
	如果接受所有 `k` 个草稿令牌，还可以“免费”从大型模型中采样额外的 `(k+1)` 个令牌（我们已计算了该分布）
	- If there was a rejection create a new rebalanced distribution at that position (`p_large - p_draft`, clamp min at 0, normalize to sum to 1) and sample the last token from it  
	如果有拒绝，则在那个位置创建一个新的重新平衡分布（ `p_large - p_draft` ，最小值限制为 0，归一化使总和为 1）并从中采样最后一个令牌

**Why this works**: Although we use the small model to propose candidates, the accept/reject rule guarantees that in expectation the sequence is distributed exactly as if we had sampled token by token from the large model. This means speculative decoding is statistically equivalent to standard autoregressive decoding — but potentially much faster, since a single large-model pass can yield up to `k+1` tokens.  
**为什么这样做有效** ：虽然我们使用小模型来提出候选，但接受/拒绝规则保证了在期望中，序列的分布与如果我们逐个从大型模型中采样令牌完全相同。这意味着投机解码在统计上等同于标准自回归解码——但可能要快得多，因为单个大型模型遍历可以产生多达 `k+1` 个令牌。

> **Note:** I recommend looking at [gpt-fast](https://github.com/meta-pytorch/gpt-fast) for a simple implementation, and the [original paper](https://arxiv.org/abs/2302.01318) for the math details and the proof of equivalence to sampling from the full model.  
> \*\*注意：\*\* 我建议查看 [gpt-fast](https://github.com/meta-pytorch/gpt-fast) 以了解简单实现，以及查看 [原始论文](https://arxiv.org/abs/2302.01318) 了解数学细节和与从完整模型采样的等价性证明。

vLLM V1 does not support the LLM draft model method, instead it implements faster—but less accurate—proposal schemes: n-gram, EAGLE [\[9\]](https://vllm.ai/blog/#ref-9), and Medusa [\[10\]](https://vllm.ai/blog/#ref-10).  
vLLM V1 不支持 LLM 草案模型方法，而是实现了更快但精度较低的提案方案：n-gram、EAGLE [\[9\]](https://vllm.ai/blog/#ref-9) 和 Medusa [\[10\]](https://vllm.ai/blog/#ref-10) 。

One-liners on each:每条一语：

- **n-gram**: take the last `prompt_lookup_max` tokens; find a prior match in the sequence; if found, propose the `k` tokens that followed that match; otherwise decrement the window and retry down to `prompt_lookup_min`  
	**n-gram** ：取最后 `prompt_lookup_max` 个标记；在序列中查找之前的匹配项；如果找到，则提出匹配项之后 `k` 个标记；否则，减小窗口并重试，直到 `prompt_lookup_min`

> **Note:** The current implementation returns `k` tokens after the first match. It feels more natural to introduce a recency bias and reverse the search direction? (i.e. last match)  
> **注意：** 当前实现返回第一个匹配项之后的 `k` 个标记。引入近期偏差并反向搜索方向是否感觉更自然？（即最后匹配项）

- **Eagle**: perform "model surgery" on the large LM—keep embeddings and LM head, replace the transformer stack with a lightweight MLP; fine-tune that as a cheap draft  
	**鹰** ：对大型语言模型进行“模型手术”——保留嵌入和语言模型头部，用轻量级 MLP 替换 Transformer 堆栈；将其微调为低成本的草稿
- **Medusa**: train auxiliary linear heads on top (embeddings before LM head) of the large model to predict the next `k` tokens in parallel; use these heads to propose tokens more efficiently than running a separate small LM  
	**美杜莎** ：在大模型的顶部（嵌入层在语言模型头部之前）训练辅助线性头部来并行预测下一个 `k` 个标记；使用这些头部比运行单独的小型语言模型更有效地提出标记

Here's how to invoke speculative decoding in vLLM using `ngram` as the draft method:  
这里是如何在 vLLM 中使用 `ngram` 作为草稿方法来调用投机解码的：

```python
from vllm import LLM, SamplingParams
 
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
 
speculative_config={
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}
 
def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", speculative_config=speculative_config)
 
    outputs = llm.generate(prompts, sampling_params)
 
if __name__ == "__main__":
    main()
```

How does this work in vLLM?  
这在 vLLM 中是如何工作的？

**Setup (during engine construction):  
设置（在引擎构建期间）：**

1. Init device: create a `drafter` (draft model, e.g., `NgramProposer`) and a `rejection_sampler` (parts of it are written in Triton).  
	初始化设备：创建一个 `drafter` （草稿模型，例如 `NgramProposer` ）和一个 `rejection_sampler` （其中一部分是用 Triton 编写的）。
2. Load model: load draft model weights (no-op for n-gram).  
	加载模型：加载草稿模型权重（对 n-gram 无操作）。

**After that in the `generate` function** (assume we get a brand new request):  
**在那之后，在 `generate` 函数** （假设我们收到一个全新的请求）：

1. Run the regular prefill step with the large model.  
	运行常规的预填充步骤，使用大型模型。
2. After the forward pass and standard sampling, call `propose_draft_token_ids(k)` to sample `k` draft tokens from the draft model.  
	在前向传播和标准采样之后，调用 `propose_draft_token_ids(k)` 从草稿模型中采样 `k` 个草稿标记。
3. Store these in `request.spec_token_ids` (update the request metadata).  
	将这些存储在 `request.spec_token_ids` 中（更新请求元数据）。
4. On the next engine step, when the request is in the running queue, add `len(request.spec_token_ids)` to the "new tokens" count so `allocate_slots` reserves sufficient KV blocks for the fwd pass.  
	在下一个引擎步骤中，当请求处于运行队列时，将 `len(request.spec_token_ids)` 添加到“新令牌”计数中，以便 `allocate_slots` 为前向传递保留足够的 KV 块。
5. Copy `spec_token_ids` into `input_batch.token_ids_cpu` to form (context + draft) tokens.  
	将 `spec_token_ids` 复制到 `input_batch.token_ids_cpu` 中，以形成（上下文 + 草稿）令牌。
6. Compute metadata via `_calc_spec_decode_metadata` (this copies over tokens from `input_batch.token_ids_cpu`, prepares logits, etc.), then run a large-model forward pass over the draft tokens.  
	通过 `_calc_spec_decode_metadata` 计算元数据（此操作将 `input_batch.token_ids_cpu` 中的标记复制过来，准备 logits 等），然后在草稿标记上运行大模型的前向传递。
7. Instead of regular sampling from logits, use the `rejection_sampler` to accept/reject left-to-right and produce `output_token_ids`.  
	不再从 logits 中进行常规采样，而是使用 `rejection_sampler` 来接受/拒绝从左到右的采样，并生成 `output_token_ids` 。
8. Repeat steps 2-7 until a stop condition is met.  
	重复步骤 2-7，直到满足停止条件。

The best way to internalize this is to fire up your debugger and step through the codebase, but this section hopefully gives you a taste for it. This as well:  
最好的方法是启动你的调试器并逐步检查代码库，但这个部分希望让你对它有所了解。同样如此：

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/4324381f0f336fc7b827e85eb9fc6790_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/ec773256a660cfca76e476b069613168_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

I've already previously hinted at the motivation behind disaggregated P/D (prefill/decode).  
我已经之前暗示过分解 P/D（预填充/解码）背后的动机。

Prefill and decode have very different performance profiles (compute-bound vs. memory-bandwidth-bound), so separating their execution is a sensible design. It gives tighter control over latency — both `TFTT` (time-to-first-token) and `ITL` (inter-token latency) — more on this in the [benchmarking](https://vllm.ai/blog/#benchmarks-and-auto-tuning---latency-vs-throughput) section.  
预填充和解码具有非常不同的性能特征（计算密集型与内存带宽密集型），因此将它们的执行分离是一种合理的设计。这可以更紧密地控制延迟——无论是 `TFTT` （首次标记时间）还是 `ITL` （标记间延迟）——更多关于这一点在 [基准测试](https://vllm.ai/blog/#benchmarks-and-auto-tuning---latency-vs-throughput) 部分。

In practice, we run `N` vLLM prefill instances and `M` vLLM decode instances, autoscaling them based on the live request mix. Prefill workers write KV to a dedicated KV-cache service; decode workers read from it. This isolates long, bursty prefill from steady, latency-sensitive decode.  
实际上，我们运行 `N` 个 vLLM 预填充实例和 `M` 个 vLLM 解码实例，根据实时请求混合自动扩展它们。预填充工作器将 KV 写入专门的 KV 缓存服务；解码工作器从中读取。这可以将长时间、突发性的预填充与稳定、延迟敏感的解码隔离开来。

How does this work in vLLM?  
这在 vLLM 中是如何工作的？

For clarity, the example below relies on `SharedStorageConnector`, a debugging connector implementation used to illustrate the mechanics.  
为了清晰起见，下面的示例依赖于 `SharedStorageConnector` ，这是一个用于说明机制的调试连接器实现。

> **Note:** Connector is vLLM's abstraction for handling the exchange of KVs between instances. Connector interface is not yet stable, there are some near-term improvements planned which will involve changes, some potentially breaking.  
> **注意：** 连接器是 vLLM 处理实例间 KVs 交换的抽象。连接器接口尚不稳定，计划进行一些近期改进，这将涉及变更，其中一些可能具有破坏性。

We launch 2 vLLM instances (GPU 0 for prefill and GPU 1 for decode), and then transfer the KV cache between them:  
我们启动了 2 个 vLLM 实例（GPU 0 用于预填充，GPU 1 用于解码），然后在这两个实例之间传输 KV 缓存：

```python
import os
import time
from multiprocessing import Event, Process
import multiprocessing as mp
 
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
 
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
 
def run_prefill(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
  sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
 
  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )
 
  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)
  llm.generate(prompts, sampling_params)
 
  prefill_done.set()  # notify decode instance that KV cache is ready
 
  # To keep the prefill node running in case the decode node is not done;
  # otherwise, the script might exit prematurely, causing incomplete decoding.
  try:
      while True:
          time.sleep(1)
  except KeyboardInterrupt:
      print("Script stopped by user.")
 
def run_decode(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
 
  sampling_params = SamplingParams(temperature=0, top_p=0.95)
 
  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )
 
  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)
 
  prefill_done.wait()  # block waiting for KV cache from prefill instance
 
  # Internally it'll first fetch KV cache before starting the decoding loop
  outputs = llm.generate(prompts, sampling_params)
 
if __name__ == "__main__":
  prefill_done = Event()
  prefill_process = Process(target=run_prefill, args=(prefill_done,))
  decode_process = Process(target=run_decode, args=(prefill_done,))
 
  prefill_process.start()
  decode_process.start()
 
  decode_process.join()
  prefill_process.terminate()
```

> **Note:** I've also experimented with `LMCache` [\[11\]](https://vllm.ai/blog/#ref-11), the fastest production-ready connector (uses NVIDIA's NIXL as the backend), but it's still at the bleeding edge and I ran into some bugs. Since much of its complexity lives in an external repo, `SharedStorageConnector` is a better choice for explanation.  
> **注意：** 我还尝试了 `LMCache` [\[11\]](https://vllm.ai/blog/#ref-11) ，这是最快的现成连接器（使用 NVIDIA 的 NIXL 作为后端），但它仍然处于前沿，我遇到了一些问题。由于其大部分复杂性都存在于外部仓库中，因此 `SharedStorageConnector` 是更好的解释选择。

These are the steps in vLLM:  
这些是 vLLM 中的步骤：

1. **Instantiation** — During engine construction, connectors are created in two places:  
	**实例化** — 在引擎构建过程中，连接器在两个地方创建：
- Inside the worker's init device procedure (under init worker distributed environment function), with role "worker".  
	在工作者的初始化设备程序中（在 init worker 分布式环境函数下），具有“worker”角色。
- Inside the scheduler constructor, with role "scheduler".  
	在调度器构造函数中，角色为“scheduler”。
1. **Cache lookup** — When the scheduler processes prefill requests from the `waiting` queue (after local prefix-cache checks), it calls connector's `get_num_new_matched_tokens`. This checks for externally cached tokens in the KV-cache server. Prefill always sees 0 here; decode may have a cache hit. The result is added to the local count before calling `allocate_slots`.  
	**缓存查找** — 当调度器处理来自 `等待` 队列的预填充请求（在本地前缀缓存检查之后），它调用连接器的 `get_num_new_matched_tokens` 。这检查 KV 缓存服务器中的外部缓存令牌。预填充在这里总是看到 0；解码可能发生缓存命中。结果在调用 `allocate_slots` 之前添加到本地计数中。
2. **State update** — The scheduler then calls `connector.update_state_after_alloc`, which records requests that had a cache (no-op for prefill).  
	**状态更新** — 然后，调度器调用 `connector.update_state_after_alloc` ，该函数记录了具有缓存的请求（对于预填充来说是无操作的）。
3. `Build metadata object` — At the end of scheduling, the scheduler calls `meta = connector.build_connector_meta`:  
	`构建元数据对象` — 在调度结束时，调度器调用 `meta = connector.build_connector_meta` ：
- Prefill adds all requests with `is_store=True` (to upload KV).  
	预填充添加所有具有 `is_store=True` （用于上传 KV）的请求。
- Decode adds requests with `is_store=False` (to fetch KV).  
	解码添加具有 `is_store=False` （用于获取 KV）的请求。
1. **Context manager** — Before the forward pass, the engine enters a KV-connector context manager:  
	**上下文管理器** — 在前向传递之前，引擎进入 KV-connector 上下文管理器：
- On enter: `kv_connector.start_load_kv` is called. For decode, this loads KV from the external server and injects it into paged memory. For prefill, it's a no-op.  
	进入时：调用 `kv_connector.start_load_kv` 。对于解码，这将从外部服务器加载 KV 并注入到分页内存中。对于预填充，这是一个空操作。
- On exit: `kv_connector.wait_for_save` is called. For prefill, this blocks until KV is uploaded to the external server. For decode, it's a no-op.  
	退出时：调用 `kv_connector.wait_for_save` 。对于预填充，这将阻塞直到 KV 上传到外部服务器。对于解码，这是一个空操作。

Here is a visual example:  
这里是一个视觉示例：

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/34ac55e898b0d98e97b80dfd650490ff_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

> \[!NOTE\] Additional notes:  
> \[注意\] 补充说明：
> 
> - For `SharedStorageConnector` "external server" is just a local file system.  
> 	对于 `SharedStorageConnector` ，“外部服务器”实际上只是一个本地文件系统。
> - Depending on configuration, KV transfers can also be done layer-by-layer (before/after each attention layer).  
> 	根据配置，KV 传输也可以逐层进行（在每个注意力层之前/之后）。
> - Decode loads external KV only once, on the first step of its requests; afterwards it computes/stores locally.  
> 	解码只在请求的第一步加载外部 KV 一次，之后它就在本地进行计算/存储。

## From UniprocExecutor to MultiProcExecutor从 UniprocExecutor 到 MultiProcExecutor

With the core techniques in place, we can now talk about scaling up.  
核心技术就位后，我们就可以讨论如何进行扩展了。

Suppose your model weights no longer fit into a single GPU's VRAM.  
假设你的模型权重不再适合放入单个 GPU 的 VRAM 中。

The first option is to shard the model across multiple GPUs on the same node using tensor parallelism (e.g., `TP=8`). If the model still doesn't fit, the next step is pipeline parallelism across nodes.  
第一个选项是将模型通过张量并行（例如， `TP=8` ）分片到同一节点上的多个 GPU 上。如果模型仍然放不下，下一步就是在节点间进行流水线并行。

> \[!NOTE\] Notes:\[!注意\] 备注：
> 
> - Intranode bandwidth is significantly higher than internode, which is why tensor parallelism (TP) is generally preferred over pipeline parallelism (PP). (It is also true that PP communicates less data than TP.)  
> 	节点间带宽远高于节点内带宽，因此张量并行（TP）通常比流水线并行（PP）更受欢迎。（实际上，PP 通信的数据量也比 TP 少。）
> - I'm not covering expert parallelism (EP) since we're focusing on standard transformers rather than MoE, nor sequence parallelism, as TP and PP are the most commonly used in practice.  
> 	由于我们专注于标准变压器而不是 MoE，因此不涉及专家并行（EP），也不涉及序列并行，因为 TP 和 PP 在实践中更常用。

At this stage, we need multiple GPU processes (workers) and an orchestration layer to coordinate them. That's exactly what `MultiProcExecutor` provides.  
在这个阶段，我们需要多个 GPU 进程（工作者）和一个协调层来管理它们。这正是 `MultiProcExecutor` 所提供的。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/99db686b3bdd7487b521232eab141a7f_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

How this works in vLLM:  
vLLM 中它是如何工作的：

1. `MultiProcExecutor` initializes an `rpc_broadcast_mq` message queue (implemented with shared memory under the hood).  
	`MultiProcExecutor` 初始化一个 `rpc_broadcast_mq` 消息队列（底层使用共享内存实现）。
2. The constructor loops over `world_size` (e.g. `TP=8 ⇒ world_size=8`) and spawns a daemon process for each rank via `WorkerProc.make_worker_process`.  
	构造函数遍历 `world_size` （例如 `TP=8 ⇒ world_size=8` ）并通过 `WorkerProc.make_worker_process` 为每个 rank 启动一个守护进程。
3. For each worker, the parent first creates a reader and writer pipe.  
	对于每个工作进程，父进程首先创建一个读取器和写入器管道。
4. The new process runs `WorkerProc.worker_main`, which instantiates a worker (going through the same "init device", "load model", etc. as in `UniprocExecutor`).  
	新流程运行 `WorkerProc.worker_main` ，该流程实例化一个工作进程（与 `UniprocExecutor` 中的“初始化设备”、“加载模型”等步骤相同）。
5. Each worker determines whether it is the driver (rank 0 in the TP group) or a regular worker. Every worker sets up two queues:  
	每个工作节点确定自己是驱动器（TP 组中的 rank 0）还是普通工作节点。每个工作节点设置两个队列：
- `rpc_broadcast_mq` (shared with the parent) for receiving work.  
	`rpc_broadcast_mq` （与父节点共享）用于接收工作。
- `worker_response_mq` for sending responses back.  
	`worker_response_mq` 用于发送响应回。
1. During initialization, each child sends its `worker_response_mq` handle to the parent via the pipe. Once all are received, the parent unblocks — this completes coordination.  
	在初始化过程中，每个子进程通过管道将它的 `worker_response_mq` 处理句柄发送给父进程。一旦所有句柄都收到，父进程解除阻塞——这完成了协调。
2. Workers then enter a busy loop, blocking on `rpc_broadcast_mq.dequeue`. When a work item arrives, they execute it (just like in `UniprocExecutor`, but now with TP/PP-specific partitioned work). Results are sent back through `worker_response_mq.enqueue`.  
	然后，工作者进入一个忙碌的循环，阻塞在 `rpc_broadcast_mq.dequeue` 上。当工作项到达时，它们执行它（就像在 `UniprocExecutor` 中一样，但现在带有 TP/PP 特定的分区工作）。结果通过 `worker_response_mq.enqueue` 发送回来。
3. At runtime, when a request arrives, `MultiProcExecutor` enqueues it into `rpc_broadcast_mq` (non-blocking) for all children workers. It then waits on the designated output rank's `worker_response_mq.dequeue` to collect the final result.  
	在运行时，当一个请求到达时， `MultiProcExecutor` 将它入队到 `rpc_broadcast_mq` （非阻塞）以供所有子工作者。然后它等待指定输出排名的 `worker_response_mq.dequeue` 以收集最终结果。

From the engine's perspective, nothing has changed — all of this multiprocessing complexity is abstracted away through a call to model executor's `execute_model`.  
从引擎的角度来看，没有任何变化——所有这些多进程复杂性都通过调用模型执行器的 `execute_model` 方法抽象化。

- In the `UniProcExecutor` case: execute\_model directly leads to calling execute\_model on the worker  
	在 `UniProcExecutor` 的情况下：execute\_model 直接导致在工作者上调用 execute\_model
- In the `MultiProcExecutor` case: execute\_model indirectly leads to calling execute\_model on each worker through `rpc_broadcast_mq`  
	在 `MultiProcExecutor` 的情况下：execute\_model 间接导致通过 `rpc_broadcast_mq` 在每个工作者上调用 execute\_model

At this point, we can run models that are as large as resources allow using the same engine interface.  
到此为止，我们可以使用相同的引擎接口运行资源允许的最大模型。

The next step is to scale out: enable data parallelism (`DP > 1`) replicating the model across nodes, add a lightweight DP coordination layer, introduce load balancing across replicas, and place one or more API servers in front to handle incoming traffic.  
下一步是扩展规模：启用数据并行性（ `DP > 1` ）将模型复制到节点上，添加轻量级 DP 协调层，引入副本之间的负载均衡，并在前面放置一个或多个 API 服务器来处理传入流量。

## Distributed system serving vLLM分布式系统服务于 vLLM

There are many ways to set up serving infrastructure, but to stay concrete, here's one example: suppose we have two H100 nodes and want to run four vLLM engines across them.  
设置服务基础设施的方法有很多，但为了具体说明，这里有一个例子：假设我们有两个 H100 节点，并想在它们上面运行四个 vLLM 引擎。

If the model requires `TP=4`, we can configure the nodes like this.  
如果模型需要 `TP=4` ，我们可以这样配置节点。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/ca8e358b4760bbfd90417d3b34dc6963_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

On the first node, run the engine in headless mode (no API server) with the following arguments:  
在第一个节点上，以无头模式（无 API 服务器）运行引擎，以下为参数：

```shell
vllm serve <model-name>
  --tensor-parallel-size 4
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-start-rank 0
  --data-parallel-address <master-ip>
  --data-parallel-rpc-port 13345
  --headless
```

and run that same command on the other node with few tweaks:  
在另一个节点上运行相同的命令，进行一些调整：

- no `--headless`
- modify DP start rank 修改 DP 开始排名

```shell
vllm serve <model-name>
  --tensor-parallel-size 4
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-start-rank 2
  --data-parallel-address <master-ip>
  --data-parallel-rpc-port 13345
```

> **Note:** This assumes networking is configured so all nodes can reach the specified IP and port.  
> **注意：** 这假定网络已配置，以便所有节点都可以访问指定的 IP 地址和端口。

How does this work in VLLM?  
这在 VLLM 中是如何工作的？

### On the headless server node在无头服务器节点上

On the headless node, a `CoreEngineProcManager` launches 2 processes (per `--data-parallel-size-local`) each running `EngineCoreProc.run_engine_core`. Each of these functions creates a `DPEngineCoreProc` (the engine core) and then enters its busy loop.  
在无头节点上，一个 `CoreEngineProcManager` 启动了 2 个进程（每个 `--data-parallel-size-local` ），每个进程运行 `EngineCoreProc.run_engine_core` 。这些函数中的每一个都创建了一个 `DPEngineCoreProc` （引擎核心），然后进入其忙碌循环。

`DPEngineCoreProc` initializes its parent `EngineCoreProc` (child of `EngineCore`), which:  
`DPEngineCoreProc` 初始化其父 `EngineCoreProc` （ `EngineCore` 的子节点），它：

1. Creates an `input_queue` and `output_queue` (`queue.Queue`).  
	创建一个 `输入队列` 和 `输出队列` (`queue.Queue`)。
2. Performs an initial handshake with the frontend on the other node using a `DEALER` ZMQ socket (async messaging lib), and receives coordination address info.  
	使用 `DEALER` ZMQ 套接字（异步消息库）与另一节点的前端进行初始握手，并接收协调地址信息。
3. Initializes DP group (e.g. using NCCL backend).  
	初始化 DP 组（例如使用 NCCL 后端）。
4. Initializes the `EngineCore` with `MultiProcExecutor` (`TP=4` on 4 GPUs as described earlier).  
	初始化 `EngineCore` ，使用 `MultiProcExecutor` （如前所述，在 4 个 GPU 上，TP=4）。
5. Creates a `ready_event` (`threading.Event`).  
	创建一个 `ready_event` （ `threading.Event` ）。
6. Starts an input deamon thread (`threading.Thread`) running `process_input_sockets(…, ready_event)`. Similarly starts an output thread.  
	启动一个输入守护线程（ `threading.Thread` ）运行于 `process_input_sockets(…, ready_event)` 。同样启动一个输出线程。
7. Still in the main thread, waits on `ready_event` until all input threads across all 4 processes (spanning the 2 nodes) have completed the coordination handshake finally executing `ready_event.set()`.  
	仍然在主线程中，等待在 `ready_event` 上，直到所有输入线程在所有 4 个进程（跨越 2 个节点）完成协调握手，最终执行 `ready_event.set()` 。
8. Once unblocked, sends a "ready" message to the frontend with metadata (e.g., `num_gpu_blocks` available in paged KV cache memory).  
	一旦解除阻塞，向前端发送带有元数据（例如，在分页的 KV 缓存内存中可用的 `num_gpu_blocks` ）的“就绪”消息。
9. The main, input, and output threads then enter their respective busy loops.  
	然后，主线程、输入线程和输出线程分别进入各自的忙碌循环。

TL;DR: We end up with 4 child processes (one per DP replica), each running a main, input, and output thread. They complete a coordination handshake with the DP coordinator and frontend, then all three threads per process run in steady-state busy loops.  
TL;DR: 我们最终得到 4 个子进程（每个 DP 副本一个），每个子进程运行一个主线程、一个输入线程和一个输出线程。它们与 DP 协调器和前端完成协调握手，然后每个进程的三个线程都在稳态忙碌循环中运行。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/db246f8d01b8bffdd44ab95900c0bd92_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

**Current steady state**:  
**当前稳态**:

- **Input thread** — blocks on the input socket until a request is routed from the API server; upon receipt, it decodes the payload, enqueues a work item via `input_queue.put_nowait(...)`, and returns to blocking on the socket.  
	**输入线程** — 在输入套接字上阻塞，直到 API 服务器路由请求；收到请求后，解码有效载荷，通过 `input_queue.put_nowait(...)` 将工作项入队，然后返回到套接字上的阻塞状态。
- **Main thread** — wakes on `input_queue.get(...)`, feeds the request to the engine; `MultiProcExecutor` runs the forward pass and enqueues results to `output_queue`.  
	**主线程** — 在 `input_queue.get(...)` 上唤醒，将请求喂给引擎； `MultiProcExecutor` 执行前向传递并将结果入队到 `output_queue` 。
- **Output thread** — wakes on `output_queue.get(...)`, sends the result back to the API server, then resumes blocking.  
	**输出线程** — 在 `output_queue.get(...)` 上唤醒，将结果发送回 API 服务器，然后继续阻塞。

**Additional mechanics**:  
**附加机制** ：

- **DP wave counter** — the system tracks "waves"; when all engines become idle they quiesce, and the counter increments when new work arrives (useful for coordination/metrics).  
	**DP 波计数器** — 系统跟踪“波”；当所有引擎处于空闲状态时，它们将静默，当有新工作到达时计数器增加（用于协调/指标）。
- **Control messages** — the API server can send more than just inference requests (e.g., aborts and utility/control RPCs).  
	**控制消息** — API 服务器可以发送的不仅仅是推理请求（例如，中止和实用/控制 RPC）。
- **Dummy steps for lockstep** — if any DP replica has work, all replicas execute a forward step; replicas without requests perform a dummy step to participate in required synchronization points (avoids blocking the active replica).  
	**用于同步的虚拟步骤** — 如果任何 DP 副本有工作，所有副本将执行前向步骤；没有请求的副本执行虚拟步骤以参与所需的同步点（避免阻塞活动副本）。

> **Note:** Lockstep clarification: this is actually only required for MoE models where the expert layers form an EP or TP group while attention layers are still DP. It's currently always done with DP - this is just because there's limited use for "built-in" non-MoE DP since you could just run multiple independent vLLMs and load-balance between them in a normal way.  
> **注意：** 步调一致说明：这实际上仅适用于 MoE 模型，其中专家层形成 EP 或 TP 组，而注意力层仍然是 DP。目前总是使用 DP - 这只是因为“内置”非 MoE DP 的使用有限，因为您可以直接运行多个独立的 vLLM，并以正常方式在它们之间进行负载均衡。

Now for the second part, what happens on the API server node?  
现在来看第二部分，API 服务器节点上会发生什么？

### On the API server node在 API 服务器节点上

We instantiate an `AsyncLLM` object (an asyncio wrapper around the LLM engine). Internally this creates a `DPLBAsyncMPClient` (data-parallel, load-balancing, asynchronous, multiprocessing client).  
我们实例化一个 `AsyncLLM` 对象（一个围绕LLM引擎的 asyncio 包装器）。内部创建一个 `DPLBAsyncMPClient` （数据并行、负载均衡、异步、多进程客户端）。

Inside the parent class of `MPClient`, the `launch_core_engines` function runs and:  
在 `MPClient` 的父类中，运行了 `launch_core_engines` 函数：

1. Creates the ZMQ addresses used for the startup handshake (as seen on the headless node).  
	创建了用于启动握手的 ZMQ 地址（如无头节点上所见）。
2. Spawns a `DPCoordinator` process.  
	启动了一个 `DPCoordinator` 进程。
3. Creates a `CoreEngineProcManager` (same as on the headless node).  
	创建一个 `CoreEngineProcManager` （与无头节点上的相同）。

Inside `AsyncMPClient` (child of `MPClient`), we:  
在 `AsyncMPClient` （ `MPClient` 的子节点）内部，我们：

1. Create an `outputs_queue` (`asyncio.Queue`).  
	创建一个 `outputs_queue` (`asyncio.Queue`)。
2. We create an asyncio task `process_outputs_socket` which communicates (through the output socket) with output threads of all 4 `DPEngineCoreProc` and writes into `outputs_queue`.  
	我们创建了一个异步任务 `process_outputs_socket` ，该任务通过输出套接字与所有 4 个 `DPEngineCoreProc` 的输出线程进行通信，并将数据写入 `outputs_queue` 。
3. Subsequently one more asyncio task `output_handler` from `AsyncLLM` reads from this queue and finally sends out information to the `create_completion` function.  
	随后，另一个异步任务 `output_handler` 从 `AsyncLLM` 中读取这个队列，并最终将信息发送到 `create_completion` 函数。

Inside `DPAsyncMPClient` we create an asyncio task `run_engine_stats_update_task` which communicates with DP coordinator.  
在 `DPAsyncMPClient` 中，我们创建了一个异步任务 `run_engine_stats_update_task` ，该任务与 DP 协调器进行通信。

The DP coordinator mediates between the frontend (API server) and backend (engine cores). It:  
DP 协调器在前端（API 服务器）和后端（引擎核心）之间进行调解。它：

- Periodically sends load-balancing info (queue sizes, waiting/running requests) to the frontend's `run_engine_stats_update_task`.  
	定期向前端发送负载均衡信息（队列大小、等待/运行中的请求）到 `run_engine_stats_update_task` 。
- Handles `SCALE_ELASTIC_EP` commands from the frontend by dynamically changing the number of engines (only works with Ray backend).  
	通过动态更改引擎数量处理来自前端 `SCALE_ELASTIC_EP` 命令（仅与 Ray 后端兼容）。
- Sends `START_DP_WAVE` events to the backend (when triggered by frontend) and reports wave-state updates back.  
	向后端发送 `START_DP_WAVE` 事件（由前端触发）并报告波形状态更新。

To recap, the frontend (`AsyncLLM`) runs several asyncio tasks (remember: concurrent, not parallel):  
总结一下，前端（ `AsyncLLM` ）运行了多个 asyncio 任务（记住：并发，不是并行）：

- A class of tasks handles input requests through the `generate` path (each new client request spawns a new asyncio task).  
	一类任务通过 `generate` 路径处理输入请求（每个新的客户端请求都会启动一个新的 asyncio 任务）。
- Two tasks (`process_outputs_socket`, `output_handler`) process output messages from the underlying engines.  
	两个任务（ `process_outputs_socket` ， `output_handler` ）处理来自底层引擎的输出消息。
- One task (`run_engine_stats_update_task`) maintains communication with the DP coordinator: sending wave triggers, polling LB state, and handling dynamic scaling requests.  
	一个任务（ `run_engine_stats_update_task` ）与 DP 协调器保持通信：发送波触发器、轮询 LB 状态和处理动态扩展请求。

Finally, the main server process creates a FastAPI app and mounts endpoints such as `OpenAIServingCompletion` and `OpenAIServingChat`, which expose `/completion`, `/chat/completion`, and others. The stack is then served via Uvicorn.  
最后，主服务器进程创建一个 FastAPI 应用，并挂载诸如 `OpenAIServingCompletion` 和 `OpenAIServingChat` 等端点，这些端点公开了 `/completion` 、 `/chat/completion` 以及其他端点。然后，通过 Uvicorn 提供该堆栈服务。

So, putting it all together, here's the full request lifecycle!  
所以，把所有内容综合起来，这就是完整的请求生命周期！

You send from your terminal:  
您从终端发送：

```curl
curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7
}'
```

What happens next:接下来会发生什么：

1. The request hits `OpenAIServingCompletion` 's `create_completion` route on the API server.  
	请求击中 API 服务器上的 `OpenAIServingCompletion` 的 `create_completion` 路由。
2. The function tokenizes the prompt asynchronously, and prepares metadata (request ID, sampling params, timestamp, etc.).  
	函数异步对提示进行分词，并准备元数据（请求 ID、采样参数、时间戳等）。
3. It then calls `AsyncLLM.generate`, which follows the same flow as the synchronous engine, eventually invoking `DPAsyncMPClient.add_request_async`.  
	然后调用 `AsyncLLM.generate` ，它遵循与同步引擎相同的流程，最终调用 `DPAsyncMPClient.add_request_async` 。
4. This in turn calls `get_core_engine_for_request`, which does load balancing across engines based on the DP coordinator's state (picking the one that has minimal score / lowest load: `score = len(waiting) * 4 + len(running)`).  
	这反过来调用 `get_core_engine_for_request` ，它根据 DP 协调器的状态（选择得分最低/负载最低的一个： `score = len(waiting) * 4 + len(running)` ）在引擎之间进行负载均衡。
5. The `ADD` request is sent to the chosen engine's `input_socket`.  
	The `ADD` 请求发送到所选引擎的 `input_socket` 。
6. At that engine:在那个引擎中：
- Input thread — unblocks, decodes data from the input socket, and places a work item on the `input_queue` for the main thread.  
	输入线程——解锁，从输入套接字解码数据，并将工作项放置在 `输入队列` 中，供主线程使用。
- Main thread — unblocks on `input_queue`, adds the request to the engine, and repeatedly calls `engine_core.step()`, enqueueing intermediate results to `output_queue` until a stop condition is met.  
	主线程——在 `输入队列` 上解锁，将请求添加到引擎中，并反复调用 `engine_core.step()` ，将中间结果入队到 `输出队列` ，直到满足停止条件。

> **Note:** Reminder: `step()` calls the scheduler, model executor (which in turn can be `MultiProcExecutor`!), etc. We have already seen this!  
> **注意：** 提醒： `step()` 调用调度器，模型执行器（它本身可以是 `MultiProcExecutor` ！），等等。我们已经看到这一点了！

- Output thread — unblocks on `output_queue` and sends results back through the output socket.  
	输出线程——在 `output_queue` 上解除阻塞，并通过输出套接字发送结果。
1. Those results trigger the `AsyncLLM` output asyncio tasks (`process_outputs_socket` and `output_handler`), which propagate tokens back to FastAPI's `create_completion` route.  
	这些结果触发 `AsyncLLM` 输出 asyncio 任务（ `process_outputs_socket` 和 `output_handler` ），它们将令牌传播回 FastAPI 的 `create_completion` 路由。
2. FastAPI attaches metadata (finish reason, logprobs, usage info, etc.) and returns a `JSONResponse` via Uvicorn to your terminal!  
	FastAPI 会将元数据（完成原因、对数概率、使用信息等）附加到通过 Uvicorn 返回到您的终端的 `JSONResponse` 上！

And just like that, your completion came back — the whole distributed machinery hidden behind a simple `curl` command!:) So much fun!!!  
就这样，您的补全结果回来了——隐藏在简单的 `curl` 命令背后的整个分布式机制！真有趣！！！

> \[!NOTE\] Additional notes:  
> \[注意\] 补充说明：
> 
> - When adding more API servers, load balancing is handled at the OS/socket level. From the application's perspective, nothing significant changes — the complexity is hidden.  
> 	当添加更多 API 服务器时，负载均衡在操作系统/套接字级别处理。从应用程序的角度来看，没有显著的变化——复杂性被隐藏了。
> - With Ray as a DP backend, you can expose a URL endpoint (`/scale_elastic_ep`) that enables automatic scaling of the number of engine replicas up or down.  
> 	使用 Ray 作为 DP 后端，您可以公开一个 URL 端点（ `/scale_elastic_ep` ），该端点可以启用自动调整引擎副本数量的上下伸缩。

## Benchmarks and auto-tuning - latency vs throughput基准测试和自动调优 - 延迟与吞吐量

So far we've been analyzing the "gas particles" — the internals of how requests flow through the engine/system. Now it's time to zoom out and look at the system as a whole, and ask: how do we measure the performance of an inference system?  
到目前为止，我们一直在分析“气体粒子”——请求如何在引擎/系统中流动的内部机制。现在，我们需要放大视角，整体审视系统，并问：我们如何衡量推理系统的性能？

At the highest level there are two competing metrics:  
在最高层面上，存在两个相互竞争的指标：

1. **Latency** — the time from when a request is submitted until tokens are returned  
	**延迟** — 从提交请求到返回标记的时间
2. **Throughput** — the number of tokens/requests per second the system can generate/process  
	**吞吐量** — 系统每秒可以生成/处理的标记/请求数量

**Latency** matters most for interactive applications, where users are waiting on responses.  
**延迟** 对于交互式应用来说最为重要，在这些应用中用户正在等待响应。

**Throughput** matters in offline workloads like synthetic data generation for pre/post-training runs, data cleaning/processing, and in general - any type of offline batch inference jobs.  
**吞吐量** 在离线工作负载中很重要，如用于预/后训练运行的合成数据生成、数据清洗/处理，以及一般而言——任何类型的离线批量推理作业。

Before explaining why latency and throughput compete, let's define a few common inference metrics:  
在解释为什么延迟和吞吐量相互竞争之前，让我们定义一些常见的推理指标：

| Metric 指标 | Definition 定义 |
| --- | --- |
| `TTFT`   (time to first token) (首次标记时间) | Time from request submission until the first output token is received   请求提交至收到第一个输出令牌的时间 |
| `ITL`   (inter-token latency) （令牌间延迟） | Time between two consecutive tokens (e.g., from token i-1 to token i)   两个连续标记之间的时间（例如，从标记 i-1 到标记 i） |
| `TPOT`   (time per output token) （每个输出标记的时间） | The average ITL across all output tokens in a request   请求中所有输出标记的平均 ITL |
| `Latency / E2E`   `延迟 / E2E`   (end-to-end latency) (端到端延迟) | Total time to process a request, i.e. TTFT + sum of all ITLs, or equivalently the time between submitting request and receiving the last output token   处理请求的总时间，即 TTFT + 所有 ITL 的总和，或者等价于提交请求与接收最后一个输出令牌之间的时间 |
| `Throughput`   吞吐量 | Total tokens processed per second (input, output, or both), or alternatively requests per second   每秒处理的令牌总数（输入、输出或两者），或者说是每秒请求数 |
| `Goodput`   `吞吐量` | Throughput that meets service-level objectives (SLOs) such as max TTFT, TPOT, or e2e latency. For example, only tokens from requests meeting those SLOs are counted   满足服务级别目标（SLOs）的吞吐量，例如最大 TTFT、TPOT 或端到端延迟。例如，只有满足这些 SLOs 的请求中的令牌被计算在内 |

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/44745068650ad4f8f94581bc51e91259_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

Here is a simplified model explaining the competing nature of these 2 metrics.  
这里是一个简化的模型，解释了这两个指标之间的竞争性。

> \[!NOTE\] Assumption: weight i/o and not KV cache i/o dominates; i.e. we're dealing with short sequences.  
> \[!注意\] 假设权重 I/O 而非 KV 缓存 I/O 占主导地位；即我们处理的是短序列。

The tradeoff becomes clear when looking at how batch size `B` affects a single decode step. As `B ↓` toward 1, ITL drops: there's less work per step and the token isn't "competing" with others. As `B ↑` toward infinity, ITL rises because we do more FLOPs per step—but throughput improves (until we hit peak perf) because weight I/O is amortized across more tokens.  
当观察批量大小 `B` 如何影响单个解码步骤时，权衡变得明显。当 `B ↓` 趋近于 1 时，ITL 下降：每步的工作量减少，并且标记不会与其他标记“竞争”。当 `B ↑` 趋近于无穷大时，ITL 上升，因为我们每步做的 FLOPs 更多——但吞吐量提高（直到达到峰值性能），因为权重 I/O 被分摊到更多的标记上。

A roofline model helps with understanding here: below a saturation batch `B_sat`, the step time is dominated by HBM bandwidth (streaming weights layer-by-layer into on-chip memory), so step latency is nearly flat—computing 1 vs 10 tokens can take a similar time. Beyond `B_sat`, the kernels become compute-bound and step time grows roughly with `B`; each extra token adds to ITL.  
屋顶线模型有助于理解这里：在饱和批量 `B_sat` 以下，步进时间主要由 HBM 带宽（逐层将权重流式传输到片上内存）决定，因此步进延迟几乎平坦——计算 1 个与 10 个标记所需时间相似。超过 `B_sat` ，内核变为计算受限，步进时间大致与 `B` 成正比；每个额外的标记都会增加 ITL。

![[公众号文章/assets/Inside vLLM Anatomy of a High-Throughput LLM Inference System/d991286b3c3b1fd4c746306ae60a4724_MD5.png]]

Note: Originally posted on Aleksa Gordic's website. 注意： 最初发布在 Aleksa Gordic 的网站 上。

> \[!NOTE\] Note: For a more rigorous treatment, we have to account for kernel auto-tuning: as `B` grows, the runtime may switch to more efficient kernels for that shape, changing the achieved performance `P_kernel`. Step latency is `t = FLOPs_step / P_kernel`, where `FLOPs_step` is the work in the step. You can see that as `P_kernel` hits `P_peak` more compute per step will directly lead to an increase in latency.  
> \[注意\] 注意：为了进行更严格的处理，我们必须考虑内核自动调整：当 `B` 增长时，运行时可能会切换到更高效的内核来处理该形状，从而改变实现的性能 `P_kernel` 。步骤延迟是 `t = FLOPs_step / P_kernel` ，其中 `FLOPs_step` 是步骤中的工作量。您可以看到，当 `P_kernel` 达到 `P_peak` 时，每步更多的计算将直接导致延迟的增加。

### How to benchmark in vLLM如何在 vLLM 中进行基准测试

vLLM provides a `vllm bench {serve,latency,throughput}` CLI that wraps vllm / benchmarks / {server,latency,throughput}.py.  
vLLM 提供了一个 `vllm bench {serve,latency,throughput}` CLI，它包装了 vllm / benchmarks / {server,latency,throughput}.py。

Here is what the scripts do:  
这里是脚本的功能：

- **latency** — uses a short input (default 32 tokens) and samples 128 output tokens with a small batch (default 8). It runs several iterations and reports e2e latency for the batch.  
	**延迟** — 使用简短输入（默认 32 个标记）并以小批量（默认 8 个）采样 128 个输出标记。它运行多次迭代，并报告批次的端到端延迟。
- **throughput** — submits a fixed set of prompts (default: 1000 ShareGPT samples) all at once (aka as `QPS=Inf` mode), and reports input/output/total tokens and requests per second across the run.  
	吞吐量（ **throughput** ）—一次性提交一组固定的提示（默认：1000 个 ShareGPT 样本），（即所谓的 `QPS=Inf` 模式），并在整个运行过程中报告每秒的输入/输出/总令牌和请求数。
- **serve** — Launches a vLLM server and simulates a real-world workload by sampling request inter-arrival times from a Poisson (or more generally, Gamma) distribution. It sends requests over a time window, measures all the metrics we’ve discussed, and can optionally enforce a server-side max concurrency (via a semaphore, e.g. limiting the server to 64 concurrent requests).  
	**启动** — 启动 vLLM 服务器并通过从泊松分布（或更一般地，伽马分布）中采样请求到达时间来模拟真实世界的工作负载。它在一个时间窗口内发送请求，测量我们讨论的所有指标，并且可以可选地强制执行服务器端最大并发数（例如，通过信号量，限制服务器最多 64 个并发请求）。

Here is an example of how you can run the latency script:  
这里是一个运行延迟脚本的示例：

```shell
vllm bench latency
  --model <model-name>
  --input-tokens 32
  --output-tokens 128
  --batch-size 8
```

> **Note:** Benchmark configs used in CI live under `.buildkite/nightly-benchmarks/tests`.  
> **注意：** CI 中使用的基准配置位于 `.buildkite/nightly-benchmarks/tests` 。

There is also an auto-tune script that drives the serve benchmark to find argument settings that meet target SLOs (e.g., "maximize throughput while keeping p99 e2e < 500 ms"), returning a suggested config.  
也有一个自动调优脚本，该脚本驱动 serve 基准测试以找到满足目标 SLO（例如，“在保持 p99 端到端百分比小于 500 毫秒的同时最大化吞吐量”）的参数设置，并返回一个建议的配置。

## Epilogue 结语

We began with the basic engine core (`UniprocExecutor`), added advanced features like speculative decoding and prefix caching, scaled up to `MultiProcExecutor` (with `TP/PP > 1`), and finally scaled out, wrapped everything in the asynchronous engine and distributed serving stack—closing with how to measure system performance.  
我们从基本的引擎核心（ `UniprocExecutor` ）开始，添加了诸如推测性解码和前缀缓存等高级功能，扩展到 `MultiProcExecutor` （TP/PP > 1），最终进行扩展，将所有内容封装在异步引擎和分布式服务栈中——最后讨论如何衡量系统性能。

vLLM also includes specialized handling that I've skipped. E.g.:  
vLLM 还包括我跳过的专用处理。例如：

- **Diverse hardware backends**: TPUs, AWS Neuron (Trainium/Inferentia), etc.  
	**多样的硬件后端** ：TPUs、AWS Neuron（Trainium/Inferentia）等。
- **Architectures/techniques**: `MLA`, `MoE`, encoder-decoder (e.g., Whisper), pooling/embedding models, `EPLB`, `m-RoPE`, `LoRA`, `ALiBi`, attention-free variants, sliding-window attention, multimodal LMs, and state-space models (e.g., Mamba/Mamba-2, Jamba)  
	**架构/技术** ： `MLA` 、 `MoE` 、编码器-解码器（例如，Whisper）、池化/嵌入模型、 `EPLB` 、 `m-RoPE` 、 `LoRA` 、 `ALiBi` 、无注意力变体、滑动窗口注意力、多模态语言模型和状态空间模型（例如，Mamba/Mamba-2、Jamba）。
- **TP/PP/SP**
- **Hybrid KV-cache logic** (Jenga), more complex sampling methods like beam sampling, and more  
	**混合 KV 缓存逻辑** （Jenga），更复杂的采样方法，如束采样等
- **Experimental**: async scheduling  
	**实验性** ：异步调度

The nice thing is that most of these are orthogonal to the main flow described above—you can almost treat them like "plugins" (in practice there's some coupling, of course).  
好处在于，这些大多数都与上述主流程正交——你几乎可以像“插件”一样对待它们（当然，实际上还是有某些耦合的）。

I love understanding systems. Having said that, the resolution definitely suffered at this altitude. In the next posts I'll zoom in on specific subsystems and get into the nitty-gritty details.  
我喜欢理解系统。话虽如此，在这个海拔高度，分辨率确实有所下降。在接下来的文章中，我将深入探讨具体的子系统，并详细阐述细节。

> \[!NOTE\] Get in touch: If you spot any errors in the post, please DM me - feel free to drop me a message on [X](https://x.com/gordic_aleksa) or [LinkedIn](https://www.linkedin.com/in/aleksagordic/) or via [anon feedback](https://docs.google.com/forms/d/1z1fEirrN2xtGxAsJvptpM7yV4ByT5SF25S-XiMPrXNA).  
> \[!注意\] 联系方式：如果您在文章中发现任何错误，请私信我 - 欢迎您在 [这里](https://x.com/gordic_aleksa) 或 [领英](https://www.linkedin.com/in/aleksagordic/) 或通过 [匿名反馈](https://docs.google.com/forms/d/1z1fEirrN2xtGxAsJvptpM7yV4ByT5SF25S-XiMPrXNA) 发送消息。

### Acknowledgments 致谢

A huge thank you to [Hyperstack](https://www.hyperstack.cloud/) for providing me with H100s for my experiments over the past year!  
非常感谢 [Hyperstack](https://www.hyperstack.cloud/) 过去一年为我提供 H100 进行实验！

Thanks to [Nick Hill](https://www.linkedin.com/in/nickhillprofile/) (core vLLM contributor, RedHat), [Kaichao You](https://github.com/youkaichao) (core vLLM contributor), [Mark Saroufim](https://x.com/marksaroufim) (PyTorch), [Kyle Krannen](https://www.linkedin.com/in/kyle-kranen/) (NVIDIA, Dynamo), and [Ashish Vaswani](https://www.linkedin.com/in/ashish-vaswani-99892181/) for reading pre-release version of this blog post and providing feedback!  
感谢 [尼克·希尔](https://www.linkedin.com/in/nickhillprofile/) （核心 vLLM 贡献者，RedHat）、 [尤卡乔·尤](https://github.com/youkaichao) （核心 vLLM 贡献者）、 [马克·萨拉菲姆](https://x.com/marksaroufim) （PyTorch）、 [凯尔·克拉内](https://www.linkedin.com/in/kyle-kranen/) （NVIDIA，Dynamo）和 [阿希什·瓦萨尼](https://www.linkedin.com/in/ashish-vaswani-99892181/) 阅读这篇博客的预发布版本并提供反馈！

References 参考文献

1. vLLM [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
2. "Attention Is All You Need" [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. "Efficient Memory Management for Large Language Model Serving with PagedAttention" [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
4. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)  
	"DeepSeek-V2：一款强大、经济、高效的混合专家语言模型" [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
5. "Jenga: Effective Memory Management for Serving LLM with Heterogeneity" [https://arxiv.org/abs/2503.18292](https://arxiv.org/abs/2503.18292)  
	"Jenga：用于服务具有异构性的LLM的有效内存管理" [https://arxiv.org/abs/2503.18292](https://arxiv.org/abs/2503.18292)
6. "Orca: A Distributed Serving System for Transformer-Based Generative Models" [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)  
	"Orca：基于 Transformer 的生成模型的分布式服务系统" [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
7. "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" [https://arxiv.org/abs/2411.15100](https://arxiv.org/abs/2411.15100)  
	"XGrammar：大型语言模型的灵活高效结构化生成引擎" [https://arxiv.org/abs/2411.15100](https://arxiv.org/abs/2411.15100)
8. "Accelerating Large Language Model Decoding with Speculative Sampling" [https://arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318)  
	"使用投机采样加速大型语言模型解码" [https://arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318)
9. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" [https://arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077)  
	"EAGLE：投机采样需要重新思考特征不确定性" [https://arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077)
10. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" [https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)  
	"美杜莎：具有多个解码头的简单 LLM 推理加速框架" [https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
11. LMCache [https://github.com/LMCache/LMCache](https://github.com/LMCache/LMCache)