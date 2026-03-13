---
title: "LLM Prefill & Decode — 完整技术图谱"
source: "https://zhuanlan.zhihu.com/p/2013203040564454281"
author:
  - "[[秋枫一名小小的DLer]]"
published:
tags:
  - "clippings"
---
![[公众号文章/assets/LLM Prefill & Decode — 完整技术图谱/69960aa91e34e45b7d1b16ac1a081103_MD5.png]]

LLM Prefill & Decode — 完整技术图谱

**LLM Inference** 两个阶段的本质定义、计算特性、瓶颈差异，以及从 KV Cache 到推测解码的全部优化方法——附进化脉络图与选型矩阵.  

| 指标 | 值 |
| --- | --- |
| Prefill 计算复杂度 | O(n²) |
| Decode 每步复杂度 | O(n) |
| KV Cache | 是桥梁 |
| 主流优化方法 | 20+ |

## 目录

**定义与原理**

- 00 Prefill & Decode — 本质定义
- 01 注意力矩阵 — 两阶段可视化
- 02 两阶段瓶颈差异分析
- 03 交互式推理过程动态演示

**优化方法**

- 04 KV Cache 优化 — 核心战场
- 05 Prefill 专项优化 — 降低 TTFT
- 06 Decode 专项优化 — 降低 TPOT
- 07 推测解码 Speculative Decoding — 一步多 Token
- 08 批处理策略对比
- 09 量化 & 模型压缩 — 底层加速

**全景视图**

- 10 优化方法进化脉络
- 11 优化方法全量对比矩阵
- 12 选型速查 — 按优化目标

---

## 00 Prefill & Decode — 本质定义

### ⚡ Prefill（预填充阶段）

**PROMPT PROCESSING · PARALLEL · COMPUTE-BOUND**

将整个输入 Prompt（N 个 Token） **一次性并行** 送入模型。每个 Token 可同时看到它之前的所有 Token（通过因果掩码）。

模型对 Prompt 中的 N 个 Token 同时计算 Q、K、V 矩阵，执行完整的 N×N 因果自注意力，并将所有 K、V 写入 KV Cache 供后续 Decode 使用。

**核心特征：** 大矩阵乘法，GPU 算力密集，Batch 越大越高效。计算量 ∝ N²（注意力矩阵），受算力限制（Compute-Bound）。

```
FLOP ≈ 2·N·d·L·12  (注意力 + FFN)
N = prompt长度，d = 隐层维度，L = 层数
吞吐瓶颈：GPU 算力峰值 (TFLOPS)
```

---

### 🔄 Decode（自回归解码阶段）

**AUTOREGRESSIVE · SEQUENTIAL · MEMORY-BOUND**

每次只处理 **1个新 Token** （当前生成的最新 Token），通过 KV Cache 复用 Prefill 阶段计算的所有历史 K/V，逐步生成输出序列，直到 EOS 或 max\_len。

新 Token 的 Q 与 KV Cache 中的所有历史 K/V 做注意力，生成下一个 Token 的概率分布，采样后追加到序列末尾。

**核心特征：** 每步仅1个 Token，计算量极小但需从显存读大量 KV Cache。受内存带宽限制（Memory-Bound），GPU 算力利用率极低（<5%）。

```cpp
FLOP/step ≈ 2·(n+t)·d·L·2  (仅当前token)
n = 历史长度，t = 当前步，显存带宽是瓶颈
吞吐瓶颈：HBM 带宽 (TB/s)
```

---

### Token 流时序可视化 — 完整推理过程

输入 Prompt（6 tokens）+ 生成输出（4 tokens）

**① PREFILL** — 一次并行处理所有 Prompt Token → 写入 KV Cache

```
[The] [quick] [brown] [fox] [jumps] [over]  → KV[0..5] 写入
```

所有 6 个 token 并行前向，计算 N×N=36 的注意力矩阵，生成第一个输出 Token

**② DECODE** — 每步仅处理 1 个新 Token，顺序自回归

```
Step 1: [The][quick][brown][fox][jumps][over] + [the]  → 生成 "lazy"
Step 2: [...][over][the]                      + [lazy] → 生成 "dog"
Step 3: [...][the][lazy]                      + [dog]  → 生成 "<eos>"
```

> 🔑 **KV Cache** 是关键桥梁：Prefill 写入历史 K/V，Decode 每步只计算新 Token 的 Q 然后与缓存 K/V 相乘，避免每步重复计算历史 Token 的 K/V，使 Decode 从 O(n²) 降至 O(n)。

## 01 注意力矩阵 — 两阶段可视化

### Prefill 注意力矩阵（因果掩码，N=6）

```
Token   The   quick  brown  fox   jumps  over
The     [1]   [ · ]  [ · ]  [ · ] [ · ] [ · ]
quick  [0.4]  [ 1 ]  [ · ]  [ · ] [ · ] [ · ]
brown  [0.3]  [0.4]  [ 1 ]  [ · ] [ · ] [ · ]
fox    [0.2]  [0.3]  [0.4]  [ 1 ] [ · ] [ · ]
jumps  [0.2]  [0.2]  [0.3]  [0.4] [ 1 ] [ · ]
over   [0.1]  [0.2]  [0.2]  [0.3] [0.4] [ 1 ]
```

所有 6 个 Token **同时计算** 。下三角可见，上三角被 mask 为 -∞。总计算量 = N²/2 = 18 个注意力元素对。

### Decode Step 3 注意力（新Token "dog" 查询全部历史）

历史 KV Cache (6+2 tokens) + 当前 Query (1 token)

```
[K:The][K:quick][K:brown][K:fox][K:jumps][K:over][K:the][K:lazy][Q→dog]
```

`Q(dog) · K[0..8]ᵀ = 注意力分数(1×9)，只需 1 行计算`

每步仅 **1 行** 注意力计算 (1×n)，而非 N×N。但需要从显存读取全部 KV Cache（O(n) 数据量）→ **内存带宽瓶颈** 。

### KV Cache 显存占用公式

```
KV Cache Size = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element

例：Llama-3-70B，FP16，4096 tokens：
2 × 80 × 8 × 128 × 4096 × 2 ≈ 84 GB（远超模型权重本身的占用量）
```

---

## 02 两阶段瓶颈差异分析

### ⚙️ Prefill — Compute-Bound

大批量大矩阵乘法，GPU 算力被充分利用（Arithmetic Intensity 高）。瓶颈在 TFLOPS（每秒浮点运算次数），而非显存带宽。

**优化方向：** 减少 FLOPs（稀疏注意力、线性注意力）、提高算力利用率（ [Flash Attention](https://zhida.zhihu.com/search?content_id=271054680&content_type=Article&match_order=1&q=Flash+Attention&zhida_source=entity) 减少 HBM 访问）、拆分并行（序列并行、Tensor 并行）。

```
Arithmetic Intensity = FLOPs / Bytes_moved
Prefill AI ≈ seq_len / 4  (AI > ridge_point → compute-bound)
A100 Ridge Point ≈ 200 FLOP/Byte
```

### 💾 Decode — Memory-Bound

每步仅1个 Token，矩阵乘法退化为矩阵-向量乘法（GEMV），GPU 大量 Tensor Core 闲置（利用率 < 5%）。瓶颈在 HBM 带宽：每步需将全部模型权重（≈70B参数×2字节=140GB）从显存读入一次，但计算量极少。

**优化方向：** KV Cache 压缩、量化减少权重大小、Continuous Batching 摊薄带宽成本、推测解码一步多 Token。

```
Decode AI = (2 × d_model) / (2 × KV_bytes + weight_bytes)
AI ≈ 1~4 FLOP/Byte (远低于 ridge_point → memory-bound)
MFU(Model FLOPs Util) 通常 < 5%
```

### 两阶段对比表

| 维度        | Prefill                   | Decode                      |
| --------- | ------------------------- | --------------------------- |
| 处理单位      | N 个 Token（并行）             | 1 个 Token（顺序）               |
| 计算复杂度     | O(N²·d)（注意力）              | O(n·d)（每步）                  |
| GPU 瓶颈    | 算力（Compute-Bound）         | 内存带宽（Memory-Bound）          |
| GPU 算力利用率 | 60–90%                    | <5%                         |
| KV Cache  | 写入（生产者）                   | 读取（消费者）                     |
| 延迟瓶颈      | TTFT（Time To First Token） | TPOT（Time Per Output Token） |
| Batch 效益  | 大 Batch 显著提升效率            | Batch 效益递减（显存限制）            |
| 扩展策略      | 增大 Batch Size             | Continuous Batching         |

---

## 03 交互式推理过程动态演示

- Token 序列状态（Prefill 阶段并行显示，Decode 阶段逐步追加）
- KV Cache 占用（随 Decode 步骤线性增长）
- GPU 算力利用率 MFU：Prefill ≈ 75%，Decode ≈ 4%
- 显存带宽压力：Prefill 为 Medium，Decode 为 HIGH

**关键数据点：**

```
Prefill: 6 tokens 并行 → KV[0..5] 已写入 | MFU ≈ 75%（Compute-Bound）
Decode Step N: Q(新token) · K[0..N+5]ᵀ → 采样下一Token | MFU ≈ 4%（Memory-Bound）
```

---

## 04 KV Cache 优化 — 核心战场

> KV Cache 是连接 Prefill 和 Decode 的桥梁，也是长上下文推理的最大显存瓶颈。优化 KV Cache 同时改善两个阶段的性能。  

### PagedAttention

**KV CACHE · 2023 | vLLM · NON-CONTIGUOUS KV · PAGE TABLE**

借鉴操作系统虚拟内存分页机制，将 KV Cache 分为固定大小的 Page（通常16个token/page）。KV 无需连续存储，通过 Page Table 映射，彻底消除内存碎片（95%→<4%碎片率）。支持 KV Cache Sharing（Prefix Caching），不同请求共享相同前缀的 KV Pages。

**效果：** 显存利用率 ↑ 70%，吞吐 ↑ 2-4×

---

### Prefix Caching / Radix Attention

**KV CACHE · 2024 | SGLang · RADIX TREE · MULTI-REQUEST SHARING**

使用基数树（Radix Tree）组织所有缓存的 KV Page，将相同前缀（System Prompt、Few-shot 示例）的 KV 在多个请求间共享。跨请求 Cache 命中时完全跳过 Prefill 阶段，TTFT 降至毫秒级。

**效果：** 相同前缀 Prefill TTFT → 0ms

---

### GQA / MQA（分组/多查询注意力）

**KV CACHE · 2024 | GROUPED QUERY ATTENTION · MULTI QUERY**

MQA：所有 Q 头共享一组 K/V，KV Cache 降至 1/h。GQA：每组 Q 头共享一组 K/V，KV Cache 降至 1/g（g=组数）。Llama 2/3、Qwen3 均采用 GQA，KV Cache 减少 4-8× 且几乎无性能损失。

**效果：** KV Cache ÷4 ~ ÷8，带宽需求大幅降低

---

### KV Cache 量化（INT8/INT4/FP8）

**KV CACHE · 2024-2025 | KV QUANTIZATION · KIVI · FLEXGEN**

对存储的 KV Cache 进行量化，从 FP16（2字节）压缩到 INT8（1字节）或 INT4（0.5字节）。研究表明 Key 对量化更敏感（通道差异大），Value 可以激进量化。KVQuant / KIVI 等方法实现 INT4 KV Cache，精度损失可控。

**效果：** KV Cache ÷2 (INT8) 或 ÷4 (INT4)

---

### KV Cache 卸载（CPU/NVMe Offload）

**KV CACHE · 2024-2025 | FLEXGEN · INFINITE LLM · CPU OFFLOAD**

将部分 KV Cache 卸载到 CPU 内存（DDR）或 NVMe SSD，突破 GPU 显存限制，支持超长上下文（1M+ tokens）。关键是调度算法：预取即将需要的 KV，隐藏 PCIe/NVLink 传输延迟。吞吐降低但上下文长度显著扩展。

**效果：** 支持 100K~1M token 上下文

---

### KV Cache 稀疏化 / 驱逐

**KV CACHE · 2024-2025 | H2O · SCISSORHANDS · SNAPKV · STREAMINGLLM**

并非所有历史 KV 都同等重要。H2O 保留"重量级"Token（高注意力分数），驱逐低重要性的历史 KV。StreamingLLM 仅保留 Attention Sink（开头4个）+ 最近 Window，支持无限长序列推理（常驻显存固定）。

**效果：** KV Cache 减少 50~80%，PPL 轻微上升

---

## 05 Prefill 专项优化 — 降低 TTFT

### Flash Attention（v1/v2/v3）

**PREFILL · 2022 | IO-AWARE TILING · SRAM REUSE · FUSED KERNEL**

将注意力计算分块（Tiling），在 SRAM（共享内存）中完成，避免将大型注意力矩阵写回 HBM（显存）。FlashAttention-2 优化并行度，FlashAttention-3 利用 Hopper 架构异步 warpgroup。从根本上改变 Prefill 的内存访问模式，HBM 访问次数从 O(N²) 降至 O(N)。

**效果：** Prefill 速度 ↑ 2-8×，显存 O(N)→O(√N)

---

### Chunked Prefill

**PREFILL · 2023-2024 | PREFILL-DECODE INTERLEAVING · LATENCY HIDING**

将长 Prompt 的 Prefill 切分为多个 Chunk，在每个 Chunk 之间插入 Decode 步骤（已有请求的生成）。解决 Prefill 霸占 GPU 导致 Decode 请求长时间等待的 "Head-of-Line Blocking" 问题，平衡 TTFT 和 TPOT，被 vLLM、SGLang 等主流框架采用。

**效果：** P99 TTFT ↓ 50%，TPOT 更稳定

---

### Prefill 分布式并行（PP + TP）

**PREFILL · 2024 | TENSOR PARALLEL · SEQUENCE PARALLEL · DISAGGREGATION**

Tensor 并行：将矩阵乘法按列/行切分到多 GPU。序列并行：将序列维度分布到多 GPU（Ring Attention）。Prefill 与 Decode 分离部署（Disaggregation）：专用 Prefill GPU 集群处理 Prompt，KV 通过 NVLink/InfiniBand 传输给 Decode 集群。

**效果：** Prefill 吞吐线性扩展（多GPU）

---

### 稀疏注意力 / 线性注意力

**PREFILL · 2023-2025 | SPARSE ATTENTION · LINEAR ATTN · MAMBA · RWKV**

减少 Prefill 的 O(N²) 注意力计算：

- 稀疏注意力（Longformer/BigBird）：局部窗口+全局Token，O(N·w)
- 线性注意力（GLA/RetNet）：核近似将 O(N²) → O(N)
- SSM 混合（Mamba/Jamba）：线性递推替代注意力，超长序列优势显著

**效果：** 超长序列 Prefill O(N²)→O(N)

---

## 06 Decode 专项优化 — 降低 TPOT

### Continuous Batching（迭代级批处理）

**DECODE · 2023 | ORCA · DYNAMIC BATCHING · ITERATION-LEVEL**

传统静态批处理：一批请求必须全部结束才能加入新请求（空位浪费）。Continuous Batching：每个 Decode 步结束时检查哪些序列已完成（输出 EOS），立即替换为新的等待请求，GPU 无空闲时间。显著提升吞吐（3-5×），是现代推理引擎（vLLM、TGI）的标配。

**效果：** 吞吐 ↑ 3-5×，GPU 利用率大幅提升

---

### CUDA Graph / CUDA Graph Capture

**DECODE · 2024 | KERNEL LAUNCH OVERHEAD · GRAPH REPLAY**

Decode 每步的计算图固定（只有输入 Token 变化），使用 CUDA Graph 将整个 Decode 步的所有 CUDA Kernel 录制为图，推理时直接 Replay，消除 Python 调度 + CUDA Kernel Launch 的开销（每步可节省 0.1-0.5ms）。vLLM、TensorRT-LLM 均默认开启。

**效果：** Decode 延迟 ↓ 10-30%

---

### 权重量化（W4A16 / W8A8 / FP8）

**DECODE · 2023-2025 | GPTQ · AWQ · SMOOTHQUANT · FP8**

减少权重字节数，直接降低每步 Decode 的内存带宽需求（带宽 ∝ 权重大小）。W4A16（AWQ/GPTQ）：权重 INT4，激活 FP16，权重大小 ÷4，带宽 ÷4。FP8（Hopper GPU 原生支持）：精度损失最小，吞吐最优。W8A8（SmoothQuant）：权重激活同时量化，适合 A100/H100。

**效果：** Decode 吞吐 ↑ 2-4×（W4 vs FP16）

---

### Prefill-Decode 分离（PD Disaggregation）

**DECODE · 2024-2025 | SPLITWISE · MOONCAKE · COMPUTE/MEMORY SPLIT**

将 Compute-Bound 的 Prefill 和 Memory-Bound 的 Decode 部署在不同 GPU 集群，各自使用最适合的硬件配置（Prefill 用多 GPU 并行，Decode 用大显存高带宽 GPU）。避免两种工作负载相互干扰。Prefill 完成后 KV 通过高速互联传输到 Decode 节点。

**效果：** Decode SLA 可控，P99 TTFT 独立优化

---

## 07 推测解码 Speculative Decoding — 一步多 Token

**核心思路：** Decode 每步只生成1个 Token 是瓶颈，但验证多个 Token 的正确性（Prefill 模式并行）比顺序生成快得多。用小模型（Draft）猜测未来 k 步 Token，再用大模型（Target）一次并行验证，接受的 Token 全部保留，从而在同等时间内生成更多 Token。

### 推测解码流程（Draft K=4 步，Accept 3 步）

```
① Draft Model（小模型）顺序猜测 k=4 步：
   [...上文] + [the] [lazy] [dog] [sat]

② Target Model（大模型）并行验证全部 k+1 个位置（Prefill 模式）：
   [...上文] + [✓the] [✓lazy] [✓dog] [✗sat→down]
   （并行验证，前3个Accept，第4个Reject并修正）

③ 结果：1次 Target 调用 → 净产出 3 个 Token（替代3次顺序 Decode）
   Accept Rate ≈ 75%，有效加速比 ≈ 3× Decode 吞吐
```

---

### Speculative Decoding（经典版）

**SPECULATIVE · 2023 | DEEPMIND · DRAFT-TARGET · LOSSLESS**

使用独立小 Draft 模型（如 68M 参数）猜测，大模型验证。基于拒绝采样的数学保证：接受概率 = min(1, p\_target/p\_draft)，结果分布与纯 Target 模型完全等价（Lossless）。关键超参：Draft 步数 k（通常 4-8），与任务和 Accept Rate 相关。

**效果：** Decode 吞吐 ↑ 2-3×（分布不变）

---

### Self-Speculative / Medusa

**SPECULATIVE · 2024 | MULTI-HEAD DRAFT · NO SEPARATE MODEL**

Medusa：在 LLM 上并行添加多个"猜测头"（Head），每个 Head 预测未来不同位置的 Token，无需独立 Draft 模型。自推测（Layer-Skip）：复用模型中间层输出作为 Draft。避免维护两个独立模型，部署简单，Accept Rate 略低于独立 Draft 模型。

**效果：** Decode ↑ 2-2.5×，无需独立Draft模型

---

### Eagle / Eagle-2 / EAGLE-3

**SPECULATIVE · 2024-2025 | FEATURE-LEVEL DRAFT · AUTO-REGRESSIVE HEAD**

EAGLE：Draft 模型在特征层（Feature Space）而非 Token 层猜测，用 Target 模型最后一层特征训练轻量 Auto-regressive Head，Accept Rate 显著高于 Medusa（≈90% vs ≈70%）。EAGLE-2 引入动态草稿树（Dynamic Draft Tree），根据 Context 自适应 k。

**效果：** Decode ↑ 3-4×，Accept Rate ≈ 90%

## 08 批处理策略对比

### Static Batching（静态批处理）

**BATCHING · 2020 | FIXED BATCH · PADDING · WASTE**

一批请求中最长序列决定 Padding 数量，最短序列完成后 GPU 空跑等待。GPU 利用率低（大量 Padding Token 的无效计算），批内新请求必须等待当前批全部完成。适合序列长度均匀的场景。

**效果：** 简单，但 GPU 利用率 30-50%

### Continuous Batching（ORCA）

**BATCHING · 2022 | ITERATION-LEVEL · DYNAMIC INSERT · NO PADDING**

每次 Decode 步后检查哪些序列完成，立即用新请求填充空位。无 Padding（PagedAttention 配合），序列完成即退出，新序列立即加入。成为 vLLM、TGI、TensorRT-LLM 等所有生产系统标配。

**效果：** 吞吐 ↑ 3-5× vs Static，接近 GPU 上限

### Chunked Prefill + Continuous Batching

**BATCHING · 2024 | PREFILL CHUNKING · MIXED BATCH · FAIR SCHEDULING**

将 Prefill 和 Decode 混合在同一 Batch 中处理（而非交替）：长 Prompt 的 Prefill 被切成小块，与多个 Decode 请求同时批处理。GPU 永远满负载（无 Prefill 独占导致 Decode 等待），P99 延迟更稳定。

**效果：** P99 TTFT/TPOT 均显著改善

### Disaggregated Prefill-Decode

**BATCHING · 2024-2025 | SPLITWISE · MOONCAKE · CLUSTER SEPARATION**

Prefill 请求路由到专用 Prefill 集群（更多 GPU 并行），Decode 请求路由到专用 Decode 集群（大显存、高带宽GPU）。KV Cache 通过 RDMA/NVLink 从 Prefill 节点传输到 Decode 节点。两类工作负载独立扩容，SLA 独立控制。

**效果：** 集群总吞吐 ↑ 2×，SLA 精确可控

## 09 量化 & 模型压缩 — 底层加速

### AWQ / GPTQ (W4A16)

**QUANT · WEIGHT-ONLY | WEIGHT INT4 · ACTIVATION FP16**

权重压缩到 INT4，激活保持 FP16。AWQ 保护关键权重通道（感知量化误差分布）；GPTQ 逐层 Hessian 二阶最优。Decode 带宽需求降至 1/4，Prefill 矩阵乘法通过 Dequant 技巧实现。

**效果：** Decode ↑ 3-4×，精度损失 <1%

### SmoothQuant / FP8

**QUANT · W+A BOTH | W8A8 · FP8 HOPPER NATIVE**

SmoothQuant：激活量化困难（异常值），将量化难度从激活"平迁"到权重。FP8（E4M3/E5M2）：H100/H800 原生支持 FP8 Tensor Core，几乎无精度损失（vs FP16），Prefill 和 Decode 都能加速。

**效果：** 吞吐 ↑ 1.5-2× vs FP16

### MLA（Multi-Head Latent Attention）

**QUANT · MLA SPECIFIC | DEEPSEEK · LOW-RANK KV COMPRESSION**

将 KV 投影到低维潜变量空间再存储（DeepSeek V2/V3 首创）。KV Cache 大小降至 1/16（vs MHA），Decode 带宽需求极低，同时保留恢复完整 KV 的能力（额外矩阵乘法，但离线于带宽瓶颈）。

**效果：** KV Cache ÷16，Decode 带宽 ↓ 10×+

## 10 优化方法进化脉络

| 年份 | 里程碑 | 核心技术 |
| --- | --- | --- |
| 2017 | Transformer | Self-Attention、KV Cache概念 |
| 2020 | GPT-3 推理优化 | KV Cache 标准化、FP16 推理 |
| 2022 | FlashAttention-1、GPTQ | IO感知注意力、权重量化起步 |
| 2023 | vLLM/PagedAttn、Spec Decode | KV分页、Continuous Batching、推测解码 |
| 2024 H1 | Prefix Cache、GQA普及、MLA | 跨请求KV共享、DeepSeek V2、FP8训练 |
| 2024 H2 | Chunked Prefill、EAGLE-2、PD Disagg | P+D分离部署、FA-3 Hopper、KV量化 |
| 2025 | Mooncake、EAGLE-3、KV Offload | KV存储分离、MegaScale、百万Token推理 |

### 三条优化主线

**Prefill 优化主线：** Transformer → FlashAttention → FA2 → FA3 → 稀疏注意力 → Chunked Prefill → 序列并行 → P/D分离集群

**KV Cache 优化主线：** 标准 KV Cache → MQA/GQA → PagedAttention → Prefix Cache → KV量化 → MLA → KV驱逐/稀疏 → CPU/NVMe Offload

**Decode 优化主线：** 逐步Decode → GPTQ量化 → Continuous Batching → 推测解码(SpecDec) → Medusa → EAGLE系列 → AWQ/W4A16 → CUDA Graph

---

## 11 优化方法全量对比矩阵

### KV Cache 管理

| 方法 | 作用阶段 | 解决问题 | 加速效果 | 代价 / 局限 | 成熟度 |
| --- | --- | --- | --- | --- | --- |
| PagedAttention | P+D | KV碎片，内存浪费 | 吞吐 ↑2-4× | 软件复杂度增加 | ⭐⭐⭐ 标配 |
| Prefix Caching | Prefill | 重复Prompt重计算 | 命中时TTFT→0 | 需缓存管理策略 | ⭐⭐⭐ 标配 |
| GQA / MQA | P+D | KV Cache太大 | KV ÷4~÷8 | 轻微精度影响 | ⭐⭐⭐ 标配 |
| KV 量化 (INT8/4) | Decode | KV带宽瓶颈 | 显存 ÷2~÷4 | 精度损失，实现复杂 | ⭐⭐ 生产中 |
| KV 驱逐 (H2O等) | Decode | 长序列KV爆炸 | KV减少50-80% | 精度损失，任务相关 | ⭐⭐ 研究阶段 |
| MLA (DeepSeek) | P+D | KV Cache显存瓶颈 | KV ÷16，带宽↓10× | 需从头训练，推理有矩阵乘 | ⭐⭐⭐ DeepSeek生产 |

### Prefill 加速

| 方法                 | 作用阶段    | 解决问题            | 加速效果          | 代价 / 局限   | 成熟度             |
| ------------------ | ------- | --------------- | ------------- | --------- | --------------- |
| FlashAttention-2/3 | Prefill | 注意力计算HBM瓶颈      | 2-8× 注意力加速    | CUDA实现复杂  | ⭐⭐⭐ 所有框架标配      |
| Chunked Prefill    | Prefill | Prefill阻塞Decode | P99 TTFT ↓50% | 调度复杂度     | ⭐⭐⭐ vLLM/SGLang |
| 稀疏注意力              | Prefill | O(N²)超长序列       | O(N²)→O(N)    | 全局信息损失    | ⭐⭐ 长文档场景        |
| Tensor/序列并行        | Prefill | 单GPU无法处理大模型     | 线性扩展          | 通信开销，复杂部署 | ⭐⭐⭐ 多GPU标配      |

### Decode 加速

| 方法 | 作用阶段 | 解决问题 | 加速效果 | 代价 / 局限 | 成熟度 |
| --- | --- | --- | --- | --- | --- |
| Continuous Batching | Decode | GPU空闲，等待 | 吞吐 ↑3-5× | 调度延迟，实现复杂 | ⭐⭐⭐ 所有框架标配 |
| CUDA Graph | Decode | Kernel启动开销 | 延迟 ↓10-30% | 图需重建（batch size变化） | ⭐⭐⭐ TRT-LLM/vLLM |
| Spec Decode (经典) | Decode | 每步仅1 Token | 吞吐 ↑2-3× | 需独立Draft模型，Accept Rate相关 | ⭐⭐ 生产逐步采用 |
| EAGLE / EAGLE-2 | Decode | 推测解码Accept Rate低 | 吞吐 ↑3-4× | 需训练Eagle头 | ⭐⭐ 快速普及中 |
| PD Disaggregation | P+D | P/D互相干扰 | 集群效率 ↑2× | KV传输开销，部署复杂 | ⭐⭐ Mooncake生产 |

### 量化 / 模型压缩

| 方法 | 作用阶段 | 解决问题 | 加速效果 | 代价 / 局限 | 成熟度 |
| --- | --- | --- | --- | --- | --- |
| FP8 (H100原生) | P+D | 计算/带宽瓶颈 | ↑1.5-2× vs FP16 | 需Hopper+架构 | ⭐⭐⭐ H100生产标配 |
| AWQ / GPTQ W4A16 | Decode主 | Decode带宽瓶颈 | Decode ↑3-4× | 量化精度损失 | ⭐⭐⭐ 广泛使用 |
| SmoothQuant W8A8 | P+D | 激活量化困难 | ↑1.5-2× | 需校准数据 | ⭐⭐⭐ 企业部署 |

---

## 12 选型速查 — 按优化目标

### 🎯 降低 TTFT（首Token延迟）

① **Prefix Caching** — 命中时 TTFT→0 ② **Chunked Prefill** — 避免 Prefill 阻塞 ③ **FlashAttention-3** — 提升 Prefill 计算效率 ④ **PD Disaggregation** — 独立扩容 Prefill 集群 ⑤ **Tensor 并行** — 多 GPU 加速单次 Prefill

### 🎯 提升吞吐（Decode TPS）

① **Continuous Batching** — 首选，GPU 无空闲 ② **Speculative Decoding (EAGLE-2)** — 一步多 Token ③ **W4A16 量化 (AWQ)** — 带宽需求 ÷4 ④ **PagedAttention** — 消除碎片，提升并发 ⑤ **GQA/MQA** — 降低 KV 显存，扩大 Batch

### 🎯 扩展上下文长度

① **MLA（训练时设计）** — KV ÷16 ② **KV 量化 INT8/INT4** — 显存 ÷2~÷4 ③ **KV Offload（CPU/NVMe）** — 突破显存限制 ④ **H2O / StreamingLLM** — 近似，无限长序列 ⑤ **Ring Attention** — 多 GPU 分布式长序列

### 🎯 降低显存 / 部署成本

① **W4A16 量化** — 模型权重 ÷4 ② **FP8 推理** — 几乎无损，H100 原生 ③ **GQA（模型设计）** — KV Cache 显存 ÷4~÷8 ④ **Prefix Cache 复用** — 减少重复计算 ⑤ **投机+小 Draft 模型** — 大模型资源更高效

还没有人送礼物，鼓励一下作者吧

编辑于 2026-03-06 11:22・广东[低至9.9元/月，即拥有 7x 24小时在线、随时响应的AI助手](https://click.aliyun.com/m/1000410524/?spu=biz%3D0%26ci%3D3673202%26si%3D645193ac-4199-4666-b21f-79f59f4e2225%26ts%3D1773408573%26zid%3D1629)

[

最多三步，即可拥有7x24小时在线、随时响应的AI...

](https://click.aliyun.com/m/1000410524/?spu=biz%3D0%26ci%3D3673202%26si%3D645193ac-4199-4666-b21f-79f59f4e2225%26ts%3D1773408573%26zid%3D1629)