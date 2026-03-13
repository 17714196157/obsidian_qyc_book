| 指标：       | **Prefix Cache Hit Rate**                                                                                                                                                | **Decode KV Cache 命中率**                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 含义        | 跨请求复用比例（APC）让请求N **复用** 请求1~N-1的缓存， **跳过** 请求N的部分Prefill计算                                                                                                               | 实际是 **存储的键值对张量**                                                                                                                                                          |
| vllm是否打印  | 有（日志打印）                                                                                                                                                                  | 无（不打印）                                                                                                                                                                    |
| 使用阶段      | Prefill阶段,**系统级跨请求指标**                                                                                                                                                   | Decode 阶段，内存访问层面                                                                                                                                                          |
| 细节区分      | APC 的缓存索引：<br>   - 只记录 "这个 token 序列曾经计算过"<br>   - 命中后，**重新应用当前位置的位置编码**<br>   - 或者：vLLM 实际上存储的是 **未加位置编码的 KV**，应用时再编码<br><br>                                            | 物理存储的 KV Cache：<br>   - 包含位置编码信息（ROPE/alibi等）<br>   - 是绝对的、不可复用的（如果位置不同）`[A,B]` 和 `[B,A]` 的注意力输出不同，因为位置编码不同                                                               |
| 处理单位      | N 个 Token（并行）                                                                                                                                                            | 1 个 Token（顺序）                                                                                                                                                             |
| 核心特征      | 大矩阵乘法，GPU 算力密集，Batch 越大越高效。计算量 ∝ N²（注意力矩阵），受算力限制（Compute-Bound）。<br><br>FLOP ≈ 2·N·d·L·12  (注意力 + FFN)<br>N = prompt长度，d = 隐层维度，L = 层数<br>吞吐瓶颈：GPU 算力峰值 (TFLOPS)<br><br> | 每步仅1个 Token，计算量极小但需从显存读大量 KV Cache。受内存带宽限制（Memory-Bound），GPU 算力利用率极低（<5%）。<br><br>FLOP/step ≈ 2·(n+t)·d·L·2  (仅当前token)<br>n = 历史长度，t = 当前步，显存带宽是瓶颈<br>吞吐瓶颈：HBM 带宽 (TB/s) |
| **优化方向**  | **减少 FLOPs（稀疏注意力、线性注意力）、提高算力利用率（Flash Attention减少 HBM 访问）、拆分并行（序列并行、Tensor 并行）。**                                                                                        | ** KV Cache 压缩、量化减少权重大小、Continuous Batching 摊薄带宽成本、推测解码一步多 Token。**                                                                                                       |
| 计算复杂度     | O(N²·d)（注意力）                                                                                                                                                             | O(n·d)（每步）                                                                                                                                                                |
| GPU 瓶颈    | 算力（Compute-Bound）                                                                                                                                                        | 内存带宽（Memory-Bound）                                                                                                                                                        |
| GPU 算力利用率 | 60–90%                                                                                                                                                                   | <5%                                                                                                                                                                       |
| KV Cache  | 写入（生产者）                                                                                                                                                                  | 读取（消费者）                                                                                                                                                                   |
| 专项优化      | TTFT（Time To First Token）                                                                                                                                                | TPOT（Time Per Output Token）                                                                                                                                               |
| Batch 效益  | 大 Batch 显著提升效率                                                                                                                                                           | Batch 效益递减（显存限制）                                                                                                                                                          |
| 扩展策略      | 增大 Batch Size                                                                                                                                                            | Continuous Batching                                                                                                                                                       |
|           |                                                                                                                                                                          |                                                                                                                                                                           |
**KV Cache 显存占用公式**
```text
KV Cache Size = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element

例：Llama-3-70B，FP16，4096 tokens：
2 × 80 × 8 × 128 × 4096 × 2 ≈ 84 GB（远超模型权重本身的占用量）
```
```markdown
# 日志示例
Engine 000: Running: 39 reqs, Waiting: 0 reqs, 
            GPU KV cache usage: 68.9%, 
            Prefix cache hit rate: 29.7%
```

## 一）APC（Automatic Prefix Caching)
**APC定义**：
 - 查询单位 ：每次请求到来时，会查询其 prompt tokens 能命中多少个已缓存的 block
- 统计方式 ：记录 `vllm:prefix_cache_queries` （查询次数）和 `vllm:prefix_cache_hits` （命中token数）
- 日志显示 ： **最近1000次查询的滑动窗口平均命中率**
- 核心机制：块级（Block-level）前缀匹配
	- a) 结构化Prompt：System Prompt单独成块，User Query新起一块
	- b) 位置无关性：System Prompt单独成块，User Query新起一块
##### 困惑问题
- 1：命中率 100% 但 TTFT 不为零（命中率 100% ≠ TTFT 为 0）？即使 100% 命中，仍有以下开销：

| 开销项             | 说明                          |
| --------------- | --------------------------- |
| **缓存查找**        | Radix Tree 遍历，O(长度)         |
| **内存拷贝**        | 从 CPU 管理的缓存拷贝到 GPU KV Cache |
| **位置编码重计算**     | 如果存储的是 base KV，需要重新应用 ROPE  |
| **后续 token 计算** | 只有前缀免计算，新 token 仍需 Prefill  |
-2 : 为什么命中率会波动，甚至之前命中的后来不命中了？
```python
GPU 显存有限 → KV Cache 容量有限 → 需要驱逐

场景：
1. 请求A：长文档X（占用 100 blocks）
2. 请求B：长文档Y（占用 100 blocks）→ 可能驱逐文档X的部分块
3. 请求C：文档X（之前能 100% 命中，现在可能 0%）

LRU 策略：最近最少使用的块被驱逐
1. 缓存是 **全局共享** 的，不是按请求隔离
2. 高并发时，热点内容可能互相驱逐
3. 长文本更容易导致缓存抖动

```
##### **RadixAttention** 算法的核心特性：

| 特性         | 说明                                           |
| ---------- | -------------------------------------------- |
| **内容寻址**   | Cache 以 **token 序列的哈希值** 为 key，而非 (请求ID, 位置) |
| **树状结构**   | 所有请求的 KV Cache 组成一棵 **Radix Tree（基数树）**      |
| **最长前缀匹配** | 新请求从根节点开始，沿树匹配最长的相同 token 序列                 |

```python
Radix Tree 结构示例：

Root
├── "You are a helpful assistant." [Block 0]
│   ├── "User: Hello" [Block 1]
│   │   └── "Assistant: Hi" [Block 2]
│   └── "User: How are you" [Block 1']
│       └── "Assistant: I'm fine" [Block 2']
│
└── "You are a coding expert." [Block 0']
    └── ...

请求C: "You are a helpful assistant. User: Hello..."
       从 Root → Block 0 → Block 1 完全命中！
       不需要知道 Block 0 是"第0个块"，只需要内容匹配
       
# 具体实现：Block 的哈希计算
# vLLM 简化逻辑（基于源码理解）
class Block:
    def compute_hash(self):
        # 只哈希当前块内的 token IDs
        # 不包含：请求ID、块位置、时间戳
        return hash(tuple(self.token_ids))
    
    def matches(self, other_block):
        return self.hash == other_block.hash
        # 位置不重要，内容相同即可
```
- 匹配必须 **从根节点开始连续**
- 一旦某个块不匹配，后续即使内容相同也无法命中
- 这是 **最长前缀匹配** 的特性，不是模糊匹配

**优化建议：**
- 固定模板用 **无意义填充** 对齐块边界
- 动态内容 **单独成块** ，不要和固定内容混在一个块里


## 二）Decode KV Cache
理论上是 100% "使用"，但不是"命中"，Decode 阶段的 KV Cache 是 **确定性访问** ，不是 **缓存查找** 
```markdown
Decode 步骤 5：
- 需要读取：Token 0-4 的 KV（之前生成的）
- 需要计算：Token 5 的 KV（新生成的）

内存访问模式：
- GPU 从 HBM 读取 Token 0-4 的 KV → 这是"访问"，不是"命中/未命中"
- 没有"未命中"概念，因为这些 KV 一定在显存里（刚算出来的）
```
但存在以下 **性能问题** ：

| 问题             | 现象                | 原因                   |
| -------------- | ----------------- | -------------------- |
| **Page Fault** | 需要分配新 block       | 显存碎片或不足              |
| **重计算**        | 重新计算之前 token 的 KV | 显存不足导致早期 KV 被驱逐（极少见） |
| **CPU-GPU 拷贝** | 从 CPU 缓存加载 KV     | 使用 CPU offloading 时  |

Decode 阶段真正瓶颈的问题 **不是命中率** ，而是 **内存带宽瓶颈** Memory-Bound：
```markdown
Decode 每步计算量：O(1) （只算一个新 token）
Decode 每步内存访问：O(N) （读取之前所有 N 个 token 的 KV）

当 N > 1000 时，95% 时间花在读取 KV Cache 上
```
这就是为什么需要：
- **KV Cache 量化** （INT8/FP8）：减少内存读取量
- **PagedAttention** ：减少内存碎片，提高连续访问效率
- **FlashAttention-Decode** ：优化内存访问模式

##### 观察到 **Decode 阶段很慢** ，可能的原因：
1. **序列太长** → KV Cache 读取量线性增长 → 使用 KV Cache 量化
2. **Batch size 太小** → GPU 计算单元空闲 → 增大 batch size 或用 continuous batching
3. **显存碎片** → PagedAttention 应该已解决，但极端情况仍存在
4. **CPU-GPU 数据传输** → 如果用了 CPU offloading，检查 `nvidia-smi` 的 PCIe 带宽


##### vLLM 默认不打印，如果观察kv cache效率
- 方法 1：启用详细日志
```bash
# 启动时增加
export VLLM_LOGGING_LEVEL=DEBUG
# 或
python -m vllm.entrypoints.openai.api_server --log-level DEBUG
```
- 方法2：使用 Prometheus 指标（如果有部署）
```python
# vLLM 暴露的 metrics 中可能有
vllm:time_to_first_token_seconds  # Prefill 时间
vllm:time_per_output_token_seconds  # Decode 每 token 时间
```



## 三条优化主线

**Prefill 优化主线：** Transformer → FlashAttention → FA2 → FA3 → 稀疏注意力 → Chunked Prefill → 序列并行 → P/D分离集群

**KV Cache 优化主线：** 标准 KV Cache → MQA/GQA → PagedAttention → Prefix Cache → KV量化 → MLA → KV驱逐/稀疏 → CPU/NVMe Offload

**Decode 优化主线：** 逐步Decode → GPTQ量化 → Continuous Batching → 推测解码(SpecDec) → Medusa → EAGLE系列 → AWQ/W4A16 → CUDA Graph

### a）Prefill 专项优化方案 — 降低 TTFT

| 方法                 | 作用阶段    | 解决问题            | 加速效果          | 代价 / 局限   | 成熟度             |
| ------------------ | ------- | --------------- | ------------- | --------- | --------------- |
| FlashAttention-2/3 | Prefill | 注意力计算HBM瓶颈      | 2-8× 注意力加速    | CUDA实现复杂  | ⭐⭐⭐ 所有框架标配      |
| Chunked Prefill    | Prefill | Prefill阻塞Decode | P99 TTFT ↓50% | 调度复杂度     | ⭐⭐⭐ vLLM/SGLang |
| 稀疏注意力              | Prefill | O(N²)超长序列       | O(N²)→O(N)    | 全局信息损失    | ⭐⭐ 长文档场景        |
| Tensor/序列并行        | Prefill | 单GPU无法处理大模型     | 线性扩展          | 通信开销，复杂部署 | ⭐⭐⭐ 多GPU标配      |

#### 1）Flash Attention（v1/v2/v3）
**PREFILL · 2022 | IO-AWARE TILING · SRAM REUSE · FUSED KERNEL**
将注意力计算分块（Tiling），在 SRAM（共享内存）中完成，避免将大型注意力矩阵写回 HBM（显存）。FlashAttention-2 优化并行度，FlashAttention-3 利用 Hopper 架构异步 warpgroup。从根本上改变 Prefill 的内存访问模式，HBM 访问次数从 O(N²) 降至 O(N)。
**效果：** Prefill 速度 ↑ 2-8×，显存 O(N)→O(√N)

#### 2）Chunked Prefill
**PREFILL · 2023-2024 | PREFILL-DECODE INTERLEAVING · LATENCY HIDING**
将长 Prompt 的 Prefill 切分为多个 Chunk，在每个 Chunk 之间插入 Decode 步骤（已有请求的生成）。解决 Prefill 霸占 GPU 导致 Decode 请求长时间等待的 "Head-of-Line Blocking" 问题，平衡 TTFT 和 TPOT，被 vLLM、SGLang 等主流框架采用。
**效果：** P99 TTFT ↓ 50%，TPOT 更稳定
#### 3） Prefill 分布式并行（PP + TP）
**PREFILL · 2024 | TENSOR PARALLEL · SEQUENCE PARALLEL · DISAGGREGATION**
Tensor 并行：将矩阵乘法按列/行切分到多 GPU。序列并行：将序列维度分布到多 GPU（Ring Attention）。Prefill 与 Decode 分离部署（Disaggregation）：专用 Prefill GPU 集群处理 Prompt，KV 通过 NVLink/InfiniBand 传输给 Decode 集群。
**效果：** Prefill 吞吐线性扩展（多GPU）
#### 4）稀疏注意力 / 线性注意力
**PREFILL · 2023-2025 | SPARSE ATTENTION · LINEAR ATTN · MAMBA · RWKV**
减少 Prefill 的 O(N²) 注意力计算：
- 稀疏注意力（Longformer/BigBird）：局部窗口+全局Token，O(N·w)
- 线性注意力（GLA/RetNet）：核近似将 O(N²) → O(N)
- SSM 混合（Mamba/Jamba）：线性递推替代注意力，超长序列优势显著
**效果：** 超长序列 Prefill O(N²)→O(N)


### b)Decode 专项优化 — 降低 TPOT

| 方法 | 作用阶段 | 解决问题 | 加速效果 | 代价 / 局限 | 成熟度 |
| --- | --- | --- | --- | --- | --- |
| Continuous Batching | Decode | GPU空闲，等待 | 吞吐 ↑3-5× | 调度延迟，实现复杂 | ⭐⭐⭐ 所有框架标配 |
| CUDA Graph | Decode | Kernel启动开销 | 延迟 ↓10-30% | 图需重建（batch size变化） | ⭐⭐⭐ TRT-LLM/vLLM |
| Spec Decode (经典) | Decode | 每步仅1 Token | 吞吐 ↑2-3× | 需独立Draft模型，Accept Rate相关 | ⭐⭐ 生产逐步采用 |
| EAGLE / EAGLE-2 | Decode | 推测解码Accept Rate低 | 吞吐 ↑3-4× | 需训练Eagle头 | ⭐⭐ 快速普及中 |
| PD Disaggregation | P+D | P/D互相干扰 | 集群效率 ↑2× | KV传输开销，部署复杂 | ⭐⭐ Mooncake生产 |

#### 1） Continuous Batching（迭代级批处理）
**DECODE · 2023 | ORCA · DYNAMIC BATCHING · ITERATION-LEVEL**
传统静态批处理：一批请求必须全部结束才能加入新请求（空位浪费）。Continuous Batching：每个 Decode 步结束时检查哪些序列已完成（输出 EOS），立即替换为新的等待请求，GPU 无空闲时间。显著提升吞吐（3-5×），是现代推理引擎（vLLM、TGI）的标配。
**效果：** 吞吐 ↑ 3-5×，GPU 利用率大幅提升
#### 2） CUDA Graph / CUDA Graph Capture
**DECODE · 2024 | KERNEL LAUNCH OVERHEAD · GRAPH REPLAY**
Decode 每步的计算图固定（只有输入 Token 变化），使用 CUDA Graph 将整个 Decode 步的所有 CUDA Kernel 录制为图，推理时直接 Replay，消除 Python 调度 + CUDA Kernel Launch 的开销（每步可节省 0.1-0.5ms）。vLLM、TensorRT-LLM 均默认开启。
**效果：** Decode 延迟 ↓ 10-30%

#### 3）权重量化（W4A16 / W8A8 / FP8）
**DECODE · 2023-2025 | GPTQ · AWQ · SMOOTHQUANT · FP8**
减少权重字节数，直接降低每步 Decode 的内存带宽需求（带宽 ∝ 权重大小）。W4A16（AWQ/GPTQ）：权重 INT4，激活 FP16，权重大小 ÷4，带宽 ÷4。FP8（Hopper GPU 原生支持）：精度损失最小，吞吐最优。W8A8（SmoothQuant）：权重激活同时量化，适合 A100/H100。
**效果：** Decode 吞吐 ↑ 2-4×（W4 vs FP16）

#### 4） Prefill-Decode 分离（PD Disaggregation）
**DECODE · 2024-2025 | SPLITWISE · MOONCAKE · COMPUTE/MEMORY SPLIT**
将 Compute-Bound 的 Prefill 和 Memory-Bound 的 Decode 部署在不同 GPU 集群，各自使用最适合的硬件配置（Prefill 用多 GPU 并行，Decode 用大显存高带宽 GPU）。避免两种工作负载相互干扰。Prefill 完成后 KV 通过高速互联传输到 Decode 节点。
**效果：** Decode SLA 可控，P99 TTFT 独立优化

### c) KV Cache 管理 优化方案

| 方法             | 作用阶段    | 解决问题         | 加速效果          | 代价 / 局限      | 成熟度            |
| -------------- | ------- | ------------ | ------------- | ------------ | -------------- |
| PagedAttention | P+D     | KV碎片，内存浪费    | 吞吐 ↑2-4×      | 软件复杂度增加      | ⭐⭐⭐ 标配         |
| Prefix Caching | Prefill | 重复Prompt重计算  | 命中时TTFT→0     | 需缓存管理策略      | ⭐⭐⭐ 标配         |
| GQA / MQA      | P+D     | KV Cache太大   | KV ÷4~÷8      | 轻微精度影响       | ⭐⭐⭐ 标配         |
| KV 量化 (INT8/4) | Decode  | KV带宽瓶颈       | 显存 ÷2~÷4      | 精度损失，实现复杂    | ⭐⭐研究阶段         |
| KV 驱逐 (H2O等)   | Decode  | 长序列KV爆炸      | KV减少50-80%    | 精度损失，任务相关    | ⭐⭐ 研究阶段        |
| MLA (DeepSeek) | P+D     | KV Cache显存瓶颈 | KV ÷16，带宽↓10× | 需从头训练，推理有矩阵乘 | ⭐⭐⭐ DeepSeek生产 |
