---
title: "【DeepSeek V4】58 页技术报告精读——从 config.json 到 model.py"
source: "https://mp.weixin.qq.com/s/iLObYwtZYqCWwRRYJVcpyA"
author:
  - "[[靳岩岩]]"
published:
created: 2026-05-14
description:
tags:
  - clippings
  - deepseek-v4
  - LLM/architecture
  - attention
  - MoE
  - FP4
---

> [!abstract] 来源信息
> 原创 靳岩岩 *2026年4月25日 03:02*

![[file-20260525162559011.webp]]


昨天 DeepSeek V4 发布，我们第一时间通过开源页面的 README 讨论了一下，基于之前对 NSA 论文和 V3.2 DSA 实现的分析，推演了 V4 的架构，写了一篇速报（第 165 期 [【DeepSeek V4开源！】1.6T 参数全家桶拆解——五大技术一次看完](https://mp.weixin.qq.com/s?__biz=MzYzMzMwNzk0NA==&mid=2247484803&idx=1&sn=f2d8f5c32750f508fcd58e3c3d0502af&scene=21#wechat_redirect)）。后来我们详细阅读了技术文档的 58 页 PDF、config.json 配置文件、model.py 推理代码，发现推演和实际有不少出入。

### 上篇推错的四点

| 上篇推测 | 实际情况 |
|----------|----------|
| 注意力架构叫"DSA2" | 实际叫 **CSA + HCA**（Compressed Sparse Attention + Heavily Compressed Attention），DSA 只是 CSA 内部的子组件 |
| Engram 被使用 | **Engram 没有被使用** |
| "五大创新" | 官方列的是 **三大创新**（混合注意力 + mHC + Muon），基础设施优化单独列章节 |
| "Mega MoE"、"FP4 Indexer" | 实际叫 Fine-Grained EP Scheme、Lightning Indexer |

> [!note] 推演的价值
> 推演的价值再快，代价是粗。这篇坐下来，把技术报告里的每个组件摊开讲。

---

## 1. CSA + HCA 混合注意力——V4 最大的创新

V4 的注意力核心是两种 **KV cache 压缩策略** 交替排列，再加一层滑动窗口兜底。

### 1.1 压缩机制

CSA 和 HCA 的压缩机制本质上是同一种操作——**学习的加权池化**：

1. 每个 token 的隐藏状态通过线性投影变成 KV cache 条目，同时生成一组"压缩权重"
2. 把相邻的 m 个 token 分成一组，用 softmax 对压缩权重做归一化，然后加权求和——m 条 KV cache 变 1 条
3. 权重不是固定的平均，是模型学出来的，还加了可学习的位置偏置（positional bias），让模型知道每个 token 在块内的相对位置

这不是简单的降采样或平均池化——每个压缩后的 KV cache 条目是原始 m 个条目的**"智能摘要"**，哪些 token 更重要、权重更高，是模型自己学会的。

> [!tip] 读代码比读论文清楚
> CSA 和 HCA 在代码里就是同一个 `Compressor` 类，区别只有一个参数 `compress_ratio`——填 4 叫 CSA，填 128 叫 HCA。技术报告里拆成两个章节、两张图、两套公式，各起了一个名字。

### 1.2 CSA 与 HCA 的区别

| 特性 | CSA | HCA |
|------|-----|-----|
| **compress_ratio** | 4（每 4 个 token 压成 1 条） | 128（每 128 个 token 压成 1 条） |
| **重叠压缩** | 有——每个压缩块融合 8 个 token 的信息 | 无 |
| **后续处理** | 压缩后跑 DSA（Lightning Indexer 选 top-k） | 压缩后全部看（dense on compressed） |
| **核心职责** | 从全局历史中精选最相关的细节 | 用极低成本扫一遍全局大意 |

**注意：DSA 没有消失**，它仍然是 V4 注意力的核心组件，训练和推理全程都在用。只不过在 V3.2 里 DSA 直接操作原始 KV cache，在 V4 里 DSA 操作的是压缩后的 KV cache——**先压缩再稀疏，两刀叠加**。

### 1.3 滑动窗口

当前 token 往前数 128 个 token 的 KV cache 原样保留，不压缩。每生成一个新 token，最老的那条被踢掉、新的补进来，像传送带一样始终只保持最近 128 条。

> [!question]- 为什么滑动窗口大小是 128？
> 因为压缩操作要凑满一个块才能执行——CSA 要凑 4 个 token，HCA 要凑 128 个。凑满之前，同块内的 token 在压缩区里是看不到的。CSA 层最多有 3 个 token 在等待区，HCA 层最多有 127 个。滑动窗口的大小取最大的那个：128，刚好覆盖 HCA 最坏情况。CSA 层用 128 有点浪费（它只需要 3 个），但统一一个数字简化了实现，而且多存的几十条开销可以忽略（128 条 × 576 字节 ≈ 0.07 MB/层）。

### 1.4 config.json 的 compress_ratios 字段

```javascript
// V4-Pro 的 compress_ratios（主模型 61 层）
128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4
```

- **128 = HCA 层**，共 31 层
- **4 = CSA 层**，共 30 层
- 排列规律：前两层 HCA，之后 CSA 和 HCA 严格交替。每一步都能同时获得"全局概览"（HCA 层）和"精选细节"（CSA 层）两种信息。

### 1.5 KV cache 压缩计算示例：50K token 上下文

V4 的 KV cache 压缩分两步：

> [!note] 第一步：维度压缩（MLA 低秩投影，2024 年 V2 首创）
> MLA（Multi-head Latent Attention）是 DeepSeek 在 2024 年 5 月发布 V2 时提出的原创架构——把所有注意力头的 KV 压缩成一份低秩的 latent vector，用的时候再投影回去。这是 DeepSeek 能撑起百万上下文的基础，没有这一层打底，后面的 CSA/HCA 再怎么压序列维度也扛不住。
>
> 具体来说：7168 维的隐藏状态通过线性投影压到 512 维（config.json: head_dim = 512, num_key_value_heads = 1），128 个 query 头共享同一份 512 维的 KV。混合精度存储（RoPE 的 64 维用 BF16，其余 448 维用 FP8），每条约 576 字节。如果不做这个投影，每条要存 7168 维 × 2 字节 = 14336 字节——**光这一步就省了 96%**。

> [!info] 第二步：序列压缩（CSA/HCA，V4 新增）
> 在 512 维的 KV cache 上，再沿序列维度压缩——把相邻的 m 条加权合并成 1 条。CSA 每 4 条压 1 条，HCA 每 128 条压 1 条。

**基础知识：KV cache 是按层存的。** 每个 token 过一层 Transformer 就产生一条 KV cache，V4-Pro 有 61 层，所以 1 个 token 要存 61 条。50000 个 token 就是 50000 × 61 = 305 万条。这就是为什么 KV cache 是推理的显存大户——token 数 × 层数 × 每条大小，三个维度相乘。

**各阶段对比（50K 上下文，61 层）：**

| 阶段 | KV cache 总量 | 省了什么 |
|------|---------------|----------|
| 标准注意力 | 40.7 GB | 基线 |
| V2 MLA 维度投影（2024） | 1.64 GB | 每条从 7168 维压到 512 维，省存储 96% |
| V3.2 DSA | 1.64 GB | 只省计算不省存储——KV cache 全存着，注意力选 top-k 不全看 |
| **V4 CSA/HCA 序列压缩** | **228 MB** | 多条压一条，再省存储 86% |

**详细计算：**

| 层类型 | 压缩后条数 | 每层大小 | 注意力实际看的条数 |
|--------|-----------|----------|-------------------|
| **CSA 层**（30 层） | 12500 条压缩 + 128 条滑动窗口 = 12628 条 | ~7.3 MB | Lightning Indexer 选 top-1024 + 滑动窗口 128 = 1152 条 |
| **HCA 层**（31 层） | 390 条压缩 + 128 条滑动窗口 = 518 条 | ~0.3 MB | 全部都看 = 518 条 |

**总计**：30 × 7.3 + 31 × 0.3 ≈ **228 MB**，相比不压缩的 1.64 GB 省了约 **86%**。

> [!important] 压缩后的 KV cache 就是最终存储的内容
> 原始的 token 级 KV 不保留（只有滑动窗口的 128 条是原始精度）。所以这既省显存又省算力——不是"压缩了算，算完还原"，是"压缩了就这么存着用"。

从 40.7 GB 到 228 MB，四代迭代压了约 **99.4%**。

> [!quote] 和 V3.2 的关键区别
> V3.2 的 DSA 是推理阶段的适配——模型训练时用的是完整注意力，推理时换成 DSA 来省算力，本质是"不改脑子，换副眼镜"。V4 在 DSA 前面加了一层 KV cache 压缩，**训练阶段就开始用**——模型从头到尾都在压缩后的 KV cache 上做注意力。DSA 本身依然在，但它的输入从原始 KV cache 变成了压缩后的 KV cache，搜索空间先被压缩砍了一刀，DSA 再砍第二刀。两刀叠加，才有了 27% FLOPs 和 10% KV cache 的效果。

---

## 2. Lightning Indexer

Lightning Indexer 只在 CSA 层（compress_ratio == 4）使用。它的工作就一件事：从压缩后的 KV cache 里选出最相关的 top-k。怎么判断"相关"？还是注意力的老办法——Query 和 Key 做点积（Q × K），分数高的就是相关的。Indexer 就是用这个分数排序，取前 1024 名。

### 2.1 工作流程

它有自己**独立的压缩器**，和 CSA 主路径的压缩器不共享。这个压缩器的流程是：

```
Hadamard 旋转 → FP4 量化
```

Hadamard 旋转是一种正交变换，作用是把向量分量的能量分散均匀，让后续的 FP4 量化误差更小。

### 2.2 config.json 对应字段

```json
"index_n_heads": 64,      // 索引头数量（Pro）
"index_head_dim": 128,    // 每个索引头的维度
"index_topk": 1024        // 选 top-1024
```

### 2.3 FP4 精度全链路

Indexer 的压缩 KV cache 用 FP4 精度存储，Query 也量化到 FP4，两者的点积（算相关性分数）全程 FP4 计算——**存的、读的、算的都是 FP4**。这是通过 QAT（量化感知训练）实现的，训练时就按 FP4 精度来优化，不是推理时硬砍的。

> [!tip] 为什么排序能用 FP4？
> 排序对精度的容忍度远高于精确计算——你只需要知道"谁排前面"，不需要知道分数差 0.001 还是 0.002。这是 Lightning Indexer 能用 FP4 的根本原因。

---

## 3. Hash Routing——和 Engram 有关系但不是 Engram

上篇写了一整节 Engram，这是最大的失误。V4 没有使用 Engram。

但 config.json 里确实有一个 Hash 相关的字段：

```json
"num_hash_layers": 3
```

这代表**前 3 层** MoE 用的是 Hash Routing（哈希路由），而不是学习的路由。

代码里 Gate 类有一个 `tid2eid` 查找表：**token ID 直接映射到 expert ID**，O(1) 查表，不需要过路由网络。PDF 原文是：

> "we replace the dense FFN layers in the initial several Transformer blocks with MoE layers that employ Hash routing (Roller et al., 2021)"

### 3.1 Hash Routing vs Engram

这个"哈希"和 Engram 论文（arXiv 2601.07372）里的哈希记忆表不是同一件事：

| 特性        | Hash Routing                      | Engram        |
| --------- | --------------------------------- | ------------- |
| **用途**    | 用哈希来固定路由——每个 token 永远去同一个专家，不需要学习 | 用哈希来存储和检索知识向量 |
| **解决的问题** | 省计算，避免负载均衡问题                      | 静态知识不该每次都过注意力 |

> [!note] 为什么前 3 层用固定路由？
> PDF 没有详细解释原因，只引用了 Roller et al., 2021（Hash Layers）。一个合理的猜测是：最底层的路由模式相对稳定，固定路由省计算还避免了负载均衡问题。

### 3.2 为什么 Engram 被放弃？

Engram 想解决的问题被 CSA 用另一种方式解决了：

- **Engram 的思路**：静态知识不该每次都过注意力，用哈希查表 O(1) 直接命中
- **CSA 的思路**：先把 KV cache 压缩到 1/4，再用 Lightning Indexer 从压缩后的 KV cache 里选 top-k——注意力还是注意力，但搜索空间被压缩 + 稀疏两刀砍到了原来的零头

效果上，CSA 把注意力成本压到 V3.2 的 27%，已经足够省了。而且 CSA 是通用方案，不挑内容类型——不管是静态事实还是需要上下文推理的内容都能处理。

Engram 的哈希查表虽然理论上 O(1) 更快，但有两个硬伤：
1. 只能处理静态知识，不挑内容类型的通用场景用不了
2. 记忆表存在 CPU DRAM 里，GPU 要用的时候得通过 PCIe 总线从 CPU 拉数据——PCIe 5.0 带宽 64 GB/s，GPU 显存带宽 3 TB/s，差了近 50 倍。省了显存但堵在总线上

在 CSA 已经把成本压到这个程度的前提下，Engram 的性价比就不够了。

---

## 4. mHC（Manifold-Constrained Hyper-Connections）

mHC 在第 37 期（[DeepSeek mHC：为什么"流形约束"是标题党——信号增益从 3000 压到 1.6 的真正原因](https://mp.weixin.qq.com/s?__biz=MzYzMzMwNzk0NA==&mid=2247483888&idx=1&sn=de3fd35a8f684836b296730687d3f7ab&scene=21#wechat_redirect)）已经详细拆解过原理，这里不重复，只补充从 V4 技术报告和代码里看到的新信息。

### 4.1 V4 的具体配置

| 参数                | 值                             |
| ----------------- | ----------------------------- |
| 残差流展宽             | 4 倍（config.json: hc_mult = 4） |
| Sinkhorn-Knopp 迭代 | 20 轮（hc_sinkhorn_iters = 20）  |

### 4.2 代码实现细节

Block 类的 `hc_pre` 方法把 4 份残差流加权合并成 1 份送入子层，`hc_post` 方法把子层输出重新展开成 4 份。对应代码里的 `hc_attn_fn` / `hc_ffn_fn`（注意力和 FFN 各一组参数）。

### 4.3 训练开销

PDF 报告 mHC 的额外 wall-time 开销只有 **6.7%**。他们用了三招压开销：
- 融合 kernel
- 选择性 recomputation（只 checkpoint 必要的中间张量）
- 调整 DualPipe 流水线让 mHC 的通信和计算重叠

---

## 5. Muon 优化器

优化器就是训练时用来更新模型权重的算法。过去几年大模型几乎都用 AdamW，V4 换成了 Muon——收敛更快、训练更稳定。

### 5.1 参数覆盖

1.6T 参数里 **99.9% 用 Muon**——注意力层的投影矩阵、MoE 专家的 FFN 权重、mHC 的混合矩阵全部用 Muon。只有 embedding、输出头、RMSNorm 这几个模块仍用 AdamW，合计不到 20 亿参数，占比 0.1%。

### 5.2 Muon vs AdamW

| 特性 | AdamW | Muon |
|------|-------|------|
| **更新方式** | 逐元素更新权重（每个参数独立算梯度方向） | 整个权重矩阵当作一个整体做正交化更新 |
| **特点** | 计算量小 | 更新方向尽量"均匀分散"，不会某些方向太猛、某些纹丝不动 |
| **代价** | — | 计算量更大，但换来的收敛速度和稳定性对 1.6T 参数的模型来说值得 |

---

## 6. MoE 具体参数

### 6.1 MoE 参数对比（V3 → V4）

|  | V3 | V4-Flash | V4-Pro |
|---|---|---|---|
| 路由专家数 | 256 | 256 | 384 |
| 激活专家数 | 8 | 6 | 6 |
| 共享专家数 | 1 | 1 | 1 |
| 专家中间维度 | 2048 | 2048 | 3072 |
| 路由打分函数 | Sigmoid | sqrtsoftplus | sqrtsoftplus |

### 6.2 激活函数变化

V3 用的是 Sigmoid 做路由打分，V4 换成了 `sqrt(softplus(x))`：

```bash
# config.json
"scoring_func": "sqrtsoftplus"
```

Softplus 是 ln(1 + e^x)，一个平滑版的 ReLU。先 softplus 保证非负，再开根号压缩动态范围。相比 Sigmoid 的优势是：不会饱和（Sigmoid 在两端梯度接近 0），梯度信号更健康。

### 6.3 SwiGLU Clamping

每个 MoE 专家内部是一个 FFN（前馈网络），用的激活函数叫 SwiGLU——这是目前大模型最常用的激活函数，LLaMA、Gemma、DeepSeek 全在用。SwiGLU 本身没有上下界限制，输出可以是任意实数，想飙多高飙多高。

V4 给它加了硬裁剪，限制在 **[-10, 10]**。原因很直接：
- 超大模型训练时，个别 token 的激活值偶尔会飙到几百，触发梯度爆炸
- V4 的专家权重是 FP4 精度——E2M1 只有 2 位指数 1 位尾数，能表示的数值范围极其有限。激活值一旦飙到几百，FP4 直接溢出

> [!warning] Clamping 不只是防梯度爆炸
> 也是让数值落在 FP4 能安全表示的范围内——不裁剪，必炸。

---

## 7. 训练细节

以下内容全部来自技术报告 PDF，上篇没有覆盖。

| 项目 | 数值 |
|------|------|
| **预训练数据量** | Pro 实际是 **33T** token，Flash 是 32T |
| **稀疏注意力启用时机** | 先用 dense attention（完整注意力）训练 **1T+ token**，再引入 sparse attention |
| **序列长度渐进扩展** | **4K → 16K → 64K → 1M**（发生在预训练过程中） |

> [!note] 稀疏注意力为什么不从头用？
> 如果从第一个 token 就用稀疏注意力，模型还没学会分配注意力就被迫做稀疏选择，效果会差。先让模型学会"怎么看"，再引入稀疏。

---

## 8. Anticipatory Routing——防训练崩溃的黑科技

### 8.1 问题

MoE 每一步训练其实有两个决策：
1. 路由网络决定"这个 token 送去哪个专家"
2. 被选中的专家拿到 token 后做实际计算

正常训练时，这两步用的是同一套参数——路由和计算看到的是同一个模型状态，同步更新。问题在于：某一步梯度更新幅度大了，路由决策突然变化，大量 token 涌入之前冷门的专家，这些专家还没准备好——**loss 突然飙升（loss spike）**。

### 8.2 解决方案

Anticipatory Routing 把这两步解耦：

| 组件 | 使用的参数 |
|------|-----------|
| **路由** | 几步之前的旧参数（"慢半拍"） |
| **专家计算** | 当前最新的参数 |

路由决策总是"慢半拍"——它看到的是几步之前的模型状态，不会被当前步的剧烈更新带偏。专家计算正常用最新参数，不受影响。

### 8.3 自动开关机制

系统实时监测 loss：

```
正常训练 → 不启用（没必要多开花销）
         ↓
检测到 loss spike → 自动启用 Anticipatory Routing 稳住路由
         ↓
loss 恢复正常 → 自动关闭
```

额外开销约 **20%**——只在 loss spike 期间付出这个代价，不是全程。

---

## 9. FP4 QAT（量化感知训练）

V4 的 FP4 不是训完之后再量化的（Post-Training Quantization），是训练过程中就让模型适应 FP4 精度的（Quantization-Aware Training, QAT）。

### 9.1 V3 FP8 vs V4 FP4 QAT

|  | V3 FP8 训练 | V4 FP4 QAT |
|---|---|---|
| **优化器存储** | FP32 | FP32 |
| **计算精度** | FP8 | FP8 |
| **额外操作** | 无 | 前向传播多一步 FP4 → FP8 模拟 |

> [!important] 关键概念澄清
> 当年 DeepSeek V3 宣传"FP8 训练"，说的是**计算精度**——矩阵乘法用 FP8 算，省算力。但优化器存储的主权重（包括动量、方差等状态）一直是 FP32 全精度，否则训练会不稳定。所谓的"低精度训练"，低的是算的精度，不是存的精度。

### 9.2 QAT 具体流程

```
FP32 主权重 → 量化到 FP4 → 反量化到 FP8 → FP8 实际计算
     ↑                                              │
     └──── 反向传播（直通估计器）←──── 梯度 ←────────┘
```

两个地方用了 FP4 QAT：
1. **MoE 专家权重**：384 个专家的权重全部走上述流程
2. **Lightning Indexer 的 QK 路径**：索引器的 Query 和 Key 也是 FP4 QAT 训练的

### 9.3 训练阶段 FP4 → FP8 无损反量化

> [!tip] 关键数学事实
> FP8 (E4M3) 比 FP4 (E2M1) 多 2 个指数位，动态范围更大。FP4 量化时会对每个 1x32 的小块计算一个 scale factor，而 FP8 量化时对每个 128x128 的大块计算一个 scale factor。只要大块内各小块 scale factor 的比值不超过一定阈值，FP8 就能完全"吸收"掉 FP4 的精细缩放信息。实测中模型权重都满足这个条件，所以训练时的 FP4 → FP8 转换**零精度损失**，而且整个 QAT 流程可以完全复用现有的 FP8 训练框架。

### 9.4 推理阶段

Blackwell 架构（B200/GB200）原生支持 FP4 × FP8 的矩阵乘法，权重 FP4 直接参与计算，激活值 FP8。

> [!note] FP4 目前的局限
> 目前 Blackwell 上 FP4 × FP8 和 FP8 × FP8 的峰值算力相同——也就是说 **FP4 目前只省显存不省算力**。PDF 说未来硬件理论上可以让 FP4 × FP8 再快 1/3，但那是下一代的事了。

---

## 10. Attention Sink

Attention Sink 存在于每一层的注意力模块里（不只是输出层），每个注意力头各有一个。

### 10.1 解决的问题

标准注意力的 softmax，所有权重加起来必须等于 1：

```cpp
weight[i] = exp(score[i]) / Σ exp(score[j])
```

哪怕所有历史 token 都和当前 Query 不相关，权重也得分出去——模型被迫"看"一些没用的 token。

### 10.2 V4 的改进

在 softmax 的分母里多加一项 exp(sink)，sink 是一个可学习参数：

```cpp
weight[i] = exp(score[i]) / (Σ exp(score[j]) + exp(sink))
```

> [!abstract] 类比理解
> 原来的 softmax 像一场投票，100% 的票必须投出去，不能弃权。sink 给了模型一个"弃权票箱"——你可以把票投进去，但投进去的票不产生任何效果。数学上，这是把注意力从概率分布（权重和 = 1）变成了次概率分布（权重和 <= 1）。

这不是 DeepSeek 独创的——OpenAI 在 GPT-5 的技术报告里也用了类似的设计。对长上下文来说几乎是必需品：百万 token 里真正相关的可能只有几百个，剩下的都是噪音。没有弃权选项，注意力必须均匀撒在噪音上；有了弃权选项，注意力可以集中在真正重要的 token 上，剩余的全部"倒进水槽"。

---

## 11. MTP（Multi-Token Prediction）

V4 延续了 V3 的 Multi-Token Prediction 训练策略：

```json
"num_nextn_predict_layers": 1
```

depth = 1，意味着模型在训练时不仅预测下一个 token，还同时预测下下一个 token。额外的预测头共享主模型的表示但有自己的输出层。

### 11.1 一次预测 2 个 token 有什么用？

好处不在推理——推理时其实还是一个一个输出的，MTP head 可以不用。好处在**训练阶段**：强迫模型在预测当前 token 的时候，内部表示就已经"想好了"下一个 token 是什么。这让中间层的表示更有前瞻性，信息密度更高。

> [!tip] 类比理解
> 考试只要求答第 1 题，你可能只想第 1 题。但要求同时答第 1 题和第 2 题，你看题目时就会多想一步——这个"多想一步"的训练信号让模型的表示质量更好。

### 11.2 为什么是 2 个不是 3 个 4 个？

这是实验结果。depth 每加 1 就多一个预测头、多一份计算量。V3 的论文里做过 ablation，depth = 1（预测 2 个）收益最高，再往上边际递减。V4 沿用了这个结论，没有加大 depth。

---

## 12. 评测数据亮点

### 硬推理

| 指标 | 成绩 |
|------|------|
| Codeforces Rating | 3206，排人类第 23 名 |
| Putnam-2025 数学证明 | 120/120 满分 |

### 编程

| 指标 | 成绩 | 对比 |
|------|------|------|
| LiveCodeBench | **93.5** | Opus 4.6: 88.8，Gemini-3.1-Pro: 91.7 |
| SWE Verified | 80.6 | 与 Opus 4.6 的 80.8 打平 |
| Terminal Bench 2.0 | 67.9 | 落后 GPT-5.4 的 75.1 |

> [!note] 编程能力总结
> 算法题能打，真实项目里还差一口气。

### 长上下文

- 超过 Gemini-3.1-Pro
- 但仍落后 Claude Opus 4.6（MRCR 1M：83.5 vs 92.9）

### 中文能力

- 中文写作赢 Gemini-3.1-Pro（62.7% vs 34.1%）
- 最难写作任务仍输 Claude Opus 4.5

### 开发者反馈

- DeepSeek 内部开发者调查：52% 的人说 V4-Pro 可以当默认编程模型

> [!quote] 长上下文差距分析
> 长上下文是 V4 进步最大的方向之一——CSA + HCA 把 KV cache 压到 10%，理论上百万 token 不再是问题。但从 MRCR 1M 的分数看，**"装得下"和"找得准"还有差距**。

---

## 13. 推理框架现状

> [!important] 生态响应比预期快
> 上期说"真正要等的是推理框架的适配"——判断没问题，但现在可以说得更具体：**vLLM 和 SGLang 都做到了 Day 0 支持**，V4 发布当天就能用。

### 13.1 各平台支持情况

| 平台 | 状态 | 说明 |
|------|------|------|
| **NVIDIA GPU + vLLM / SGLang** | ✅ Day 0 支持 | 原生支持 FP4+FP8 混合精度，Blackwell 架构效果最好，NVFP4 kernel 让 MoE 专家通信量减少 4 倍，推理吞吐提升最多 1.8 倍 |
| **官方 PyTorch 脚本** | ✅ 可用 | 裸 PyTorch，默认 model-parallel=8，没有 continuous batching，适合验证和实验 |
| **MLX（Mac 用户）** | ❌ 暂无适配 | V4 的异构注意力是硬骨头，Apple Silicon 没有 FP4 原生指令。**装得下，跑不了** |
| **llama.cpp** | 🔄 进行中 | 社区有兴趣但 GGUF 量化版还没完全就绪 |
| **昇腾 950PR** | ✅ 全面适配 | 华为官方宣布昇腾超节点系列全面支持，V4-Pro 20ms TPOT / 4700 TPS，V4-Flash 10ms / 16000 TPS（8K 上下文） |
| **寒武纪 MLU** | ✅ Day 0 适配 | Torch-MLU-Ops 适配 Compressor 和 mHC，BangC 写稀疏/压缩注意力 kernel，支持 5D 混合并行 |

### 13.2 硬件方案更新

**Mac Studio M3 Ultra**：256GB 内存确实装得下 160GB 的 V4-Flash 权重，但 MLX 还没有适配 DeepseekV4ForCausalLM 架构，Apple Silicon 没有 FP4 原生指令。**目前状态：装得下，跑不了。**

**两台 DGX Spark 直连**：vLLM 已经 Day 0 支持 V4，并且支持 `--tensor-parallel-size 2` 双卡并行。两台 Spark 直连共 256GB，装 160GB 的 V4-Flash 绰绰有余，而且 Spark 是 Blackwell 架构原生支持 FP4。**目前状态：从"跑不了"变成"值得试"。**

### 13.3 技术挑战

V4 的架构对推理框架的挑战不小，但 vLLM 和 SGLang 在 Day 0 就全部解决了：

- **异构 KV cache**：每层结构不同，PagedAttention 的基本假设被打破
- **压缩状态缓冲**：凑满 m 个 token 才能压一次
- **Lightning Indexer 的动态 top-k 选择**：需要专门 kernel

> [!note] 尚未实现的优化
> PDF 里描述的 **On-Disk KV Cache**（三种磁盘缓存策略，用于共享前缀复用）是目前唯一还没被第三方框架实现的生产级优化。

---

## 14. 整体回顾

| 组件 | 上篇怎么写的 | 实际是什么 |
|------|-------------|-----------|
| 注意力架构 | DSA2（五合一） | CSA + HCA（两种压缩注意力交替排列 + 滑动窗口） |
| 知识检索 | Engram 哈希记忆 | 没有使用 Engram，前 3 层用 Hash Routing |
| 稀疏注意力索引 | FP4 Indexer | Lightning Indexer（FP4 QAT） |
| MoE 推理优化 | Mega MoE | Fine-Grained EP Scheme（PDF 正式名称） |
| 官方列的三大创新 | 五大创新 | 混合注意力 + mHC + Muon |

> [!quote] 速报和精读的差别
> 上篇的大方向没错——V4 确实是在注意力稀疏化、MoE 推理优化、训练稳定性三个方向上同时推进。但名字叫错了，Engram 写多了，创新分类也不对。
>
> 速报和精读的差别，就在于**"坐下来看代码"**这一步。config.json 的 `compress_ratios` 字段、model.py 的 `Compressor` 类、PDF 第 4 章的训练细节——这些东西不读原文是猜不到的。

---

## 附录：本文引用的原始资料

### DeepSeek 官方

- 技术报告 PDF（58 页）：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf
- config.json：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
- model.py（推理代码）：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py
- generate.py（生成脚本）：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/generate.py
- V4-Pro 模型主页：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- V4-Flash 模型主页：https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

### 推理框架适配

- vLLM DeepSeek V4 支持：https://vllm.ai/blog/deepseek-v4
- SGLang DeepSeek V4 文档：https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4

### 国产芯片适配

- 华为昇腾全面支持 V4：https://www.ithome.com/0/943/029.htm
- 寒武纪 Day 0 适配 V4：https://www.ithome.com/0/943/018.htm
