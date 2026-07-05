---
title: "DeepSpec：用草稿模型提升80%推理速度"
source: "https://mp.weixin.qq.com/s/mWLq0k2fLQPN2xATB1d1Rw"
author:
  - "[[数字生命体验官]]"
published:
created: 2026-07-05
github: "https://github.com/deepseek-ai/DeepSpec"
description:
tags:
  - clippings
  - deepseek
  - speculative-decoding
  - deepspec
  - draft-model
  - 推理加速
  - DSpark
  - Gemma4
---

## 概述

10 小时前，DeepSeek 开源了 **DeepSpec** 项目，短短十小时 860+ stars。

它不是一个新的大模型，也不是常见的微调框架。它做的事情更窄，也更工程化：围绕 **Speculative Decoding（推测解码）**，训练一个能帮目标大模型提前打草稿的 draft model，从而提升推理速度。

> [!abstract] 一句话总结
> DeepSpec 训练的是一个"会提前猜 token 的草稿模型"，让大模型每次 forward 尽量多推进几个 token。

---

## 1. 为什么大模型推理慢

普通自回归生成是**串行**的：

```
生成第 1 个 token → 才能生成第 2 个 → 才能生成第 3 个 → ...
每一步都要跑一次目标模型。
```

模型越大，这个过程越贵。

### 1.1 Speculative Decoding 的核心思路

> [!tip] 核心想法
> 很多后续 token 其实不难猜。能不能先让一个便宜的模型猜几步，再让大模型一起检查？

**举例：**

```
当前上下文：DeepSpec 可以用来

draft model 先猜：训练 / 草稿 / 模型 / 系统
target model 一次 forward 检查
→ 接受前三个，拒绝"系统"，重新采样"。"
→ 最终输出：训练 / 草稿 / 模型 / 。
```

target model 一次检查推进了 4 个位置里的 3 个。只要接受长度足够高，整体推理速度就能上来。

### 1.2 关键规则：连续前缀接受

> [!important] 接受必须是连续前缀
> - 第 1 个 token 接受了，才看第 2 个
> - 第 2 个接受了，才看第 3 个
> - **中间任何一个被拒绝，后面的草稿全部作废**

所以 draft model 要学的不是"随便猜一个看起来像的 token"，而是让**概率分布尽量贴近 target model**。贴得越近，接受率越高。

### 1.3 为什么结果不会跑偏

Speculative Decoding 保证输出分布与 target model 对齐的机制：

```
draft model 提出候选 token
       ↓
target model 计算目标分布下的概率
       ↓
   接受？──是──→ 直接输出
       ↓否
   从残差分布重新采样 → 保持与 target model 对齐
```

> [!quote] 核心保证
> 如果 token 被拒绝，不是简单取 target model 最大概率 token，而是**从残差分布里重新采样**。这样整体采样结果仍然和 target model 对齐。

---

## 2. DeepSpec 在解决什么问题

DeepSpec 把 speculative decoding 里最麻烦的一整段链路做成了**工程框架**：

```
数据准备 → target cache → draft model 训练 → speculative decoding 评测
```

### 2.1 仓库结构

```
deepspec/
├── scripts/data/     数据准备
├── train.py          训练入口
├── eval.py           评测入口
├── config/           DSpark / DFlash / Eagle3 配置
└── deepspec/         模型、训练器、评测器实现
```

### 2.2 支持的模型类型

| 维度 | 支持内容 |
|------|---------|
| **Draft Model 类型** | DSpark、DFlash、Eagle3 |
| **目标模型** | Qwen3、Gemma4 |

> [!note] 本文重点
> 作者重点分析的是 DSpark，尤其是 Gemma4 版本里的 `Gemma4DSparkModel`。

---

## 3. 草稿模型到底是什么

> [!abstract] Draft Model 定位
> 可以把 draft model 理解成一个**"提案器"**。它每轮提出多个 token，target model 负责审核。

| Draft Model | 替代小模型 |
|-------------|-----------|
| 只是提案，输出必须经 target model 验证 | 直接决定输出，不经审核 |

---

## 4. DeepSpec 的训练链路

### 4.1 Target Cache 机制

DeepSpec 的训练前置步骤比较重：

```
先跑 target model → 缓存训练样本的内部状态 → 训练时直接读取
```

**缓存字段：**

| 字段 | 说明 |
|------|------|
| `input_ids` | 输入 token IDs |
| `loss_mask` | 损失计算掩码 |
| `target_hidden_states` | target model 中间层隐状态 |
| `target_last_hidden_states` | target model 最后一层隐状态 |

> [!warning] 存储成本
> 默认 Qwen3-4B 配置下，完整 target cache 大约能到 **38 TB**。换成更大模型或缓存更多层 hidden states，存储压力还会继续上来。
>
> **DeepSpec 不是随便在笔记本上跑的项目，它更像是面向多卡机器、大容量高速存储的实验框架。**

### 4.2 工程取舍

| 好处 | 代价 |
|------|------|
| 训练 loop 更轻 | cache 极大，磁盘压力高 |
| draft model 直接读取 target hidden states | 需要大容量高速存储 |

---

## 5. Gemma4DSparkModel 的核心设计

> [!important] 核心思想
> target model 先留下几层"思考轨迹"，draft model 沿着这些轨迹往后猜。

### 5.1 五大关键模块

| 模块 | 说明 | 默认配置 |
|------|------|---------|
| **1. Embedding & lm_head** | 从 target model 拷贝并冻结，输入输出词表空间天然对齐 | 冻结 |
| **2. Target Feature Projection** | 多层 target hidden states 拼接 → fc 投回 draft hidden size → norm | `target_layer_ids = [5, 17, 29, 41, 46]` |
| **3. DSpark Backbone** | 采样 anchor position，每个 anchor 训练一个 draft block | `block_size = 7`<br>`num_anchors = 512` |
| **4. Markov Head** | 基于前一个 token 的低秩 Markov bias，加强局部 token 转移建模 | `markov_rank = 256`<br>`markov_head_type = "vanilla"` |
| **5. Confidence Head** | 预测每个 draft token 被 target model 接受的概率，评测时可设 threshold 提前停止 | 可调 |

> [!tip] Target Layer 选择
> `target_layer_ids = [5, 17, 29, 41, 46]` 说明不是只看 target model 最后一层，而是**取多层信息**。

> [!note] Markov Head 开关
> 把 `markov_rank` 设为 0 即可关闭 Markov bias。DFlash 配置基本就是这个方向。

### 5.2 一次训练 Forward 的流程

```
1. 从 loss_mask 有效位置里采样 anchor
2. 每个 anchor 构造一个 draft block
3. block 第一个位置放 anchor token
4. 后面位置放 mask token
5. draft backbone 结合 target hidden context 生成 hidden states
6. lm_head 输出 draft logits
7. Markov head 修正 logits
8. confidence head 预测接受概率
9. 计算 loss
```

> [!example] Draft Block 的巧妙之处
> 不是完全自回归地一个 token 一个 token 生成，而是用 **mask token 搭出一块"草稿区域"**。模型要在这块区域里**并行学习**后续 token。
>
> 这和最终推理时的目标一致：**一次提案多个 token**。

---

## 6. 训练 Loss 为什么不只用 CE

普通语言模型用 cross entropy 基本够用。但 speculative decoding 关心的不只是"预测 token 对不对"，还关心**draft model 的概率分布和 target model 有多接近**。

> [!important] 为什么分布对齐重要
> 接受概率跟两个分布的接近程度有关。只优化 CE = 只优化 next token 分类 ≠ 优化分布对齐。

### 6.1 Loss 组成

| Loss 项 | 作用 | 默认权重 |
|---------|------|---------|
| **CE Loss** | next token 分类 | `ce_loss_alpha = 0.1` |
| **L1 Distribution Loss** | 概率分布对齐 | `l1_loss_alpha = 0.9` |
| **Confidence Loss** | 接受概率预测 | `confidence_head_alpha = 1.0` |

此外还有 `loss_decay_gamma = 4.0`，用来**降低 block 后部 token 的权重**。越往后的 token，越依赖前面的 token 全部被接受，训练噪声也更大。

> [!quote] 权重解读
> CE 权重 0.1 vs L1 权重 0.9 —— 说明它**更重视分布对齐，而不是单纯 next token 分类**。

---

## 7. 评测时应该看哪些指标

DeepSpec 的 `eval.py` 不是简单算 loss，而是**真的跑 speculative decoding**。

### 7.1 内置评测任务

```
gsm8k, math500, aime25, humaneval, mbpp,
livecodebench, mt-bench, alpaca, arena-hard-v2
```

### 7.2 关键指标

| 指标 | 含义 | 诊断价值 |
|------|------|---------|
| **#propose** | 每轮 draft 提案长度（如 `7.00+1`，7 是 draft token，+1 是 target 额外采样） | 提案能力 |
| **accept_len** | 平均每轮接受多少 token | 越高 = 每次审核推进越多 |
| **verify_rate** | target model 验证频率 | 接受长度越高，verify rate 越低 |
| **accept_rate@position** | 位置级接受率 | **最适合定位问题的指标** |

### 7.3 位置级接受率诊断

| 现象 | 原因 | 建议 |
|------|------|------|
| `accept_rate@0` 很低 | 第一个 token 都不稳，draft 和 target 分布没对齐 | 检查训练、调 loss 权重 |
| `accept_rate@0` 很高但 `accept_rate@5` 很低 | 前面还行，block 后部开始掉 | 调小 block size、调 loss 权重、调整 target layer 选择、调整 confidence threshold |

---

## 8. H800 上完整跑通流程

以 Gemma4 DSpark 为例，单机 8 卡流程：

### 8.1 环境准备

```bash
conda create -n deepspec python=3.11 -y
conda activate deepspec
python -m pip install -r requirements.txt
python -m pip install "sglang[all]"
```

### 8.2 多卡环境

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
```

### 8.3 完整流程

```
下载切分数据 → 启动 SGLang target server → 生成 target answers → 构建 target cache → 训练 draft model → 评测
```

**Step 1: 下载数据**
```bash
python scripts/data/download_and_split.py \
  --dataset-name mlabonne/open-perfectblend \
  --test-size 0.05 \
  --train-output-path train_datasets/perfectblend_train.jsonl \
  --test-output-dir eval_datasets \
  --skip-existing
```

**Step 2: 启动 Target Model 服务**
```bash
bash scripts/data/launch_sglang_server.sh
```

**Step 3: 生成 Target Answers**
```bash
python scripts/data/generate_train_data.py \
  --model google/gemma-4-12B-it \
  --server-address 127.0.0.1:30000 127.0.0.1:30001 ... \
  --concurrency 32 \
  --temperature 0.7 \
  --max-tokens 4096 \
  --input-file-path train_datasets/perfectblend_train.jsonl \
  --output-file-path train_datasets/gemma4_12b/perfectblend_train_regen.jsonl
```

**Step 4: 构建 Target Cache**
```bash
export target_cache_dir=${HOME}/.cache/deepspec/gemma4_12b_target_cache

python scripts/data/prepare_target_cache.py \
  --config config/dspark/dspark_gemma4_12b.py \
  --train-data-path train_datasets/gemma4_12b/perfectblend_train_regen.jsonl \
  --output-dir ${target_cache_dir} \
  --local-batch-size 16
```

**Step 5: 训练 Draft Model**
```bash
python train.py \
  --config config/dspark/dspark_gemma4_12b.py \
  --opts "data.target_cache_path=${target_cache_dir}" \
  --opts "train.local_batch_size=1" \
  --opts "train.global_batch_size=512"
```

> Checkpoint 默认写到：`~/checkpoints/deepspec/dspark_block8_gemma4_12b/step_*`

**Step 6: 评测**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python eval.py \
  --target_name_or_path google/gemma-4-12B-it \
  --draft_name_or_path ${HOME}/checkpoints/deepspec/dspark_block8_gemma4_12b/step_latest \
  --max-new-tokens 2048 \
  --temperature 1.0 \
  --confidence-threshold 0.0
```

### 8.4 OOM 调参指南

如果 OOM，优先降这些参数：

| 参数 | 影响 |
|------|------|
| `train.local_batch_size` | 单卡 batch size |
| `model.num_anchors` | 每条样本 anchor 数量 |
| `data.max_length` | 序列最大长度 |
| `model.target_layer_ids` 数量 | 缓存的 target layer 层数 |

> [!warning] 磁盘提醒
> 如果磁盘吃紧，先别急着跑完整数据。**target cache 是这个项目里最容易被低估的成本。**
>
> H800 80GB 上，建议先用小数据集冒烟。确认 cache manifest、训练 loss、`accept_rate@0` 都正常，再扩大数据规模。

---

## 9. 开发者经验总结

### 9.1 三个诊断问题

| 问题 | 看什么指标 | 说明 |
|------|-----------|------|
| **1. Draft model 的第一个 token 稳不稳？** | `accept_rate@0` | 第一个 token 都接不住，后面不用谈 |
| **2. Block 后部掉得快不快？** | `accept_rate@position` | 前面高后面断崖式下降 → 调 block size、loss 权重、confidence 策略 |
| **3. Cache 成本能不能承受？** | 磁盘空间 | target layer 选得越多，cache 越大。训练前先算磁盘 |

### 9.2 核心价值

> [!success] DeepSpec 的真正价值
> 不在于它"又训练了一个小模型"。它真正补齐的是 **speculative decoding 的工程闭环**：
>
> - 怎么准备 target 数据
> - 怎么缓存 target hidden states
> - 怎么训练 draft model
> - 怎么评估真实接受率
> - 怎么比较不同 draft 方案

> [!quote] 总结
> DeepSpec 不是单点模型实现，而是一套面向 speculative decoding 的**实验工作台**。它把"提升推理速度"这件事拆成了几个可以被观察、被训练、被比较的环节。对开发者来说，这比只给一个论文公式或者单个 benchmark 数字更有价值。
