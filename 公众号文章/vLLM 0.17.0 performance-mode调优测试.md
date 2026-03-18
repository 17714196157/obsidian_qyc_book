---
title: "vLLM 0.17.0 performance-mode调优测试"
source: "https://mp.weixin.qq.com/s/e2KyVCOpi_wgbeCtLx0TmQ"
author:
  - "[[IanSun]]"
published:
tags:
  - "clippings"
---
原创 IanSun *2026年3月9日 18:11*

本文用于测试和记录vLLM 0.17.0版本所更新的--performance-mode选项，根据更新日志所示。该功能的目的是为了简化vLLM性能调优的各类选项，帮助用户能够更快的启用vLLM服务。

![[公众号文章/assets/vLLM 0.17.0 performance-mode调优测试/27769a76164fec10fdba0fc90f1cb443_MD5.webp]]

介绍

![[公众号文章/assets/vLLM 0.17.0 performance-mode调优测试/c50ac9201e8d0e8d6c0fd0715355a244_MD5.webp]]

**`-performance-mode {balanced,interactivity,throughput}` 。**

默认值为 `balanced` ，这保留了现有的行为。  
为了启动并演示该功能，设置 `interactivity` （交互性）模式会在小批量（step-1 最多 32）时捕获细粒度的 CUDA 图；而 `throughput` （吞吐量）模式会将默认批量大小限制加倍。

未来我们可以使用此标志来：

- 更改调度器行为，以专门处理批量大小或批量平滑度
- 捕获更多的 CUDA 图
- 决定在无法在运行时选择的预言机（oracles）中使用哪些内核，例如 Triton 与 DeepGEMM MoE
- 增加 API 服务器数量或流间隔，以减少 CPU 瓶颈
- 启用有助于降低延迟或提高吞吐量的融合操作

  

测试

本次测试环境为RHEL10.1，Nvidia L40S 驱动为CUDA 13.1，使用的vLLM版本为0.17.0，测试模型为Qwen3.5-9B

依次测试balanced,interactivity,throughput三个模式下的性能，测试工具采用vLLM自带的bench测试工具。

```bash
通过HF CLI下载Qwen3.5-9B到本地export MODEL_LOCAL_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" #替换为你的模型路径vllm serve $MODEL_LOCAL_PATH     --served-model-name qwen3.5-9b     --kv-cache-dtype fp8     --max-model-len 32768     --gpu-memory-utilization 0.85     --host 0.0.0.0     --port 8000  --performance-mode throughput
```

![[公众号文章/assets/vLLM 0.17.0 performance-mode调优测试/47ede638c706093694deeef1519b7138_MD5.webp]]

```bash
测试命令，依次调整--request-rate 从1到15增加测试压力
vllm bench serve     --model qwen3.5-9b     --tokenizer $MODEL_LOCAL_PATH     --base-url http://127.0.0.1:8000/v1     --endpoint /completions     --dataset-name random     --random-input-len 1024     --random-output-len 128     --num-prompts 100     --request-rate 5.0
```

**以下是NVIDIA L40S** 在 **RHEL 10** 环境下运行 **Qwen3.5-9B** (1024 输入 / 128 输出) 在不同performance-mode下的测试数据。

### vLLM 性能模式横向对比总表

| **模式 (Performance Mode)** | **RPS (请求速率)** | **Mean TTFT (首字延迟)** | **Mean TPOT (逐字延迟)** | **Total Throughput (总吞吐)** | **状态评估** |
| --- | --- | --- | --- | --- | --- |
| **Throughput (吞吐优先)** | 1.0 | 149.45 ms | 27.24 ms | 1114.01 tok/s | 极致轻载 |
|  | 5.0 | 272.53 ms | 49.54 ms | 4762.52 tok/s | 高效运行 |
|  | 8.0 | 498.50 ms | 70.48 ms | 6297.70 tok/s | 负载适中 |
|  | 10.0 | 12100.51 ms\* | 90.61 ms | 3874.78 tok/s | *瞬时阻塞* |
|  | 12.0 | **918.58 ms** | 75.88 ms | **7246.99 tok/s** | **巅峰性能** |
|  | 15.0 | 1374.72 ms | 77.28 ms | **7471.55 tok/s** | **极限吞吐** |
|  |  |  |  |  |  |
| **Balanced (平衡模式)** | 1.0 | 791.55 ms | 29.09 ms | 1074.53 tok/s | 较稳定 |
|  | 5.0 | 4127.27 ms | 77.22 ms | 4435.99 tok/s | 出现排队 |
|  | 8.0 | 523.46 ms | 70.19 ms | 6240.77 tok/s | 黄金平衡点 |
|  | 10.0 | 751.02 ms | 74.98 ms | 6755.68 tok/s | 满载峰值 |
|  | 12.0 | 4057.28 ms | 78.69 ms | 5899.17 tok/s | 高负载受控 |
|  | 15.0 | 4851.45 ms | 80.67 ms | 5860.61 tok/s | 极限抗压 |
|  |  |  |  |  |  |
| **Interactivity (交互优先)** | 1.0 | **153.03 ms** | 27.38 ms | 1113.91 tok/s | **极致响应** |
|  | 5.0 | 4172.92 ms | 76.88 ms | 4439.84 tok/s | 调度开销 |
|  | 8.0 | 532.01 ms | 70.45 ms | 6232.69 tok/s | 交互甜点位 |
|  | 10.0 | 761.43 ms | 75.11 ms | 6745.65 tok/s | 稳定运行 |
|  | 12.0 | 4087.60 ms | 78.88 ms | 5883.19 tok/s | 抗压稳健 |
|  | 15.0 | 4851.16 ms | 80.75 ms | 5858.36 tok/s | 极限受控 |

测试数据分析

最高吞吐冠军：Throughput 模式 (15.0 RPS)

在 15.0 RPS 时达到了 7471.55 tok/s 的总吞吐量，比其他模式高出约 27%。这归功于它将 max\_num\_batched\_tokens 提升到了 4096，极大压榨了 L40S 的并行计算能力。

极致响应冠军：Interactivity 模式 (1.0 RPS)

153.03 ms 的平均首字延迟（TTFT）提供了最丝滑的用户体感。更重要的是，在之前的 1.0 RPS 测试中，它的 P99 极值（258ms）远优于默认模式（9284ms），证明了它在低并发下的极致稳定性。

异常波动点分析 (Throughput 10.0 RPS)

在 10.0 RPS 下出现的 12秒延迟属于明显的瞬时阻塞或计算图重录制。有趣的是，在更重负载的 12.0 和 15.0 RPS 下，系统反而通过更大规模的 Batch 合并恢复了正常响应（约 1秒 TTFT）。

黄金建议 (Best RPS)

如果你希望兼顾响应速度和吞吐量，8.0 RPS 是全模式下的“性能拐点”：延迟保持在 500ms 左右，吞吐量维持在 6200 tok/s 以上。

不同模式下的日志分析

在 vLLM v0.17.0 中开启 --performance-mode throughput（吞吐量模式），是将引擎配置为“大推力”状态。它与之前测试的 interactivity 和 balanced 模式在底层参数上有显著的区别。

throughput 模式主要启用了以下选项和行为：

1\. Chunked Prefill 容量翻倍 (核心差异)

日志显示：

Chunked prefill is enabled with max\_num\_batched\_tokens=4096.

对比分析：在 balanced 和 interactivity 模式中，这个值通常是 2048。

作用：吞吐量模式允许系统在单个批次（Batch）中处理更多的输入 Token（Prefill）。这意味着显卡可以更饱和地运行张量核心（Tensor Cores），减少了处理长文本或大并发时的调度次数。虽然这会稍微增加首字延迟（TTFT），但极大提升了总吞吐量。

  

2\. 编译范围（Compile Range）扩大

日志显示：

Cache the graph of compile range (1, 4096) for later use

Compiling a graph for compile range (1, 4096) takes 17.25 s

对比分析：交互模式下由于 max\_num\_batched\_tokens 较小，编译范围通常只到 2048。

作用：它为更大的计算规模预做了 torch.compile。可以看到这导致启动时的编译时间大幅增加（17.25s 对比之前的 1s 左右），这就是为了换取运行时的极致速度。

  

3\. CUDA Graph 捕获策略：深度覆盖大 Batch

日志显示：

Capturing CUDA graphs (decode, FULL): 100%... 51/51

Graph capturing finished in 7 secs, took 1.63 GiB

对比分析：

interactivity: 捕获了 60 个 Decode 图，侧重小 Batch 的线性覆盖。

throughput: 捕获了 51 个图。

逻辑：它不再纠结于 Batch Size = 1, 2, 3 这种细碎的捕获，而是侧重于覆盖更大的并发数（Power of 2 步进）。它的目标是确保当同时有几十个甚至上百个请求并发时，能命中最高效的 GPU 计算路径。

  

4\. 调度器倾向：最大化利用率 (Batching-First)

虽然日志中没有直接打印，但 throughput 模式在引擎内部调整了以下逻辑：

增加等待窗口：调度器会更积极地合并请求。如果一瞬间来了多个请求，它会尽量把它们塞进同一个推理循环中执行。

减少抢占（Preemption）：为了维持高吞吐，它会尽量避免为了新请求而抢占正在生成的旧请求，以保持显存总线的高效利用。

vLLM 性能模式参数配置对比表

| **核心参数** | **Interactivity (交互优先)** | **Balanced (平衡模式)** | **Throughput (吞吐优先)** |
| --- | --- | --- | --- |
| **max\_num\_batched\_tokens** | **2048** (限制单次吞吐) | 2048 | **4096** (大推力) |
| **CUDA Graph Capture Sizes** | **线性密集捕获 (1,2,3...32)** | 稀疏步进捕获 (1,2,4,8...) | 稀疏步进捕获 (1,2,4,8...) |
| **Graph Capture Count** | **76 (Mixed) / 60 (Decode)** | 51 (Mixed) / 35 (Decode) | 51 (Mixed) / 51 (Decode) |
| **Graph Memory Usage** | **1.94 GiB** (显存换延迟) | 1.38 GiB | 1.63 GiB |
| **Compilation Range** | \[1, 2048\] | \[1, 2048\] | **\[1, 4096\]** |
| **Batching Wait Time** | **趋近于 0** (即刻出发) | 微小延迟 (寻找 Batch) | **允许更长延迟** (合并大 Batch) |

1\. Interactivity (交互模式)：极致消灭“抖动”

启用逻辑：通过线性捕获 Batch Size 从 1 到 32 的所有 CUDA Graphs。

起到的作用：在低并发（1-5个请求）时，无论请求如何交错，都能精准命中预制的计算图快照。

实战表现：在 1.0 RPS 下跑出的 153ms (Mean) 和 258ms (P99) 就是这一参数的直接战果。它消除了 Python 层的调度开销，让首字延迟极度平滑。

  

2\. Balanced (平衡模式)：防灾崩溃的“保险丝”

启用逻辑：设置适中的 batched\_tokens (2048)，开启 Asynchronous scheduling（异步调度）。

起到的作用：它在内部开启了 Chunked Prefill。这意味着当一个长文本请求（Prefill）正在处理时，它不会完全霸占 GPU，而是切成小块，允许其他用户的生成请求（Decoding）插队通过。

实战表现：在 15.0 RPS 的高压下，它依然能维持 5860 tok/s 的输出，而没有像默认模式那样直接锁死到 13 秒延迟。

  

3\. Throughput (吞吐模式)：开启“暴力”推理

启用逻辑：将 max\_num\_batched\_tokens 翻倍至 4096，并扩大 torch.compile 编译范围。

起到的作用：允许 GPU 在一次时钟循环中处理更多的 Token。对于 L40S 这种拥有强大 Tensor Core 的显卡，Batch 越大，每瓦特算力的输出效率越高。

实战表现：成功刷出了 7471.55 tok/s 的全场最高分。代价是启动时需要长达 17 秒的深度编译（Compile Range 4096），但这对于长时间运行的服务器来说非常划算。

  

小结

追求“毫秒级响应” (Best for Chat)：选择 Interactivity。它牺牲了大约 0.6GB 显存来消除 99% 的系统抖动，让单用户响应稳在 150ms 级别。

追求“生产环境抗压” (Best for API)：选择 Balanced。它是一个全能选手，在 8.0 RPS 时性能表现最完美，延迟与吞吐比例最协调。

追求“数据处理效率” (Best for Batch Offline)：选择 Throughput。它是为了压榨 L40S 硬件上限而生，适合离线翻译、文档总结等不要求瞬时回复的场景。

\--performance-mode在我们需要简单性能调优的情况下，可以很好的帮助我们实现自优化的目标，但如果要解决一些特定的场景需求则还需要手工进行参数优化调整。

  

注：以上均为测试环境下所测得数据，可根据自身情况进行对应测试获得更符合你环境的实际数据。

  
ref:https://github.com/vllm-project/vllm/pull/34936
https://github.com/vllm-project/vllm/releases
