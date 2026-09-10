---
title: "【面试必考】LayerNorm为什么被RMSNorm全面取代？从0讲透归一化+手撕LLaMA同款代码"
url: "https://www.bilibili.com/video/BV1owTx6pEqM?spm_id_from=333.788.recommend_more_video.7&trackid=web_related_0.router-related-2589621-l95s2.1784622017548.878&vd_source=d0a50f3d250eed1f7d1546f70041c66b"
bvid: "BV1owTx6pEqM"
cid: "39613235254"
author: "古希腊掌管代码的神"
upload_date: "2026-07-04"
subtitle_lang: "中文"
created: "2026-07-21"
tags:
  - clippings
  - bilibili
  - LayerNorm
  - RMSNorm
  - 归一化
  - LLaMA
  - Transformer
  - 面试
---

<iframe src="https://player.bilibili.com/player.html?aid=116854769190023&bvid=BV1owTx6pEqM&cid=39613235254&page=1&autoplay=0" scrolling="no" border="0" frameborder="no" framespacing="0" allow="fullscreen; picture-in-picture" allowfullscreen="true" style="height:100%;width:100%; aspect-ratio: 16 / 9;"> </iframe>

## 概述

全行业把用了 7 年的 LayerNorm 集体换掉，就为了少算一个平均数——这不是抠门，这是大模型架构里**最划算的一笔交易**。

> [!abstract] 一句话总结
> LayerNorm 的功劳几乎全来自**"缩放不变性"**（把幅度拧回标准），而**"平移归零"**贡献微乎其微。RMSNorm 砍掉多余步骤，参数减半、提速 24.7%、效果一分不掉。

---

## 一、为什么大模型必须归一化

### 深层网络的数值失控

大模型是几十层 Block 摞起来的，信号要一层一层往上传。每一层都会把数值放大或缩小一点点，30 层承下来就是**指数级**：

```
不加控制：
    才到第 4 层就已经放大 20 多倍
        ↓
    要么爆炸成天文数字（梯度爆炸）
    要么缩到没有梯度（梯度消失）
        ↓
    训练直接崩
```

### 解决方案：音量校准器

> [!tip] 朴素想法
> 在每层入口装一个**"音量校准器"**，把数值拧回标准范围再进门。

**现代大模型用 Pre-Norm：** 每个 Block 里装两次——进 Attention 前一次，进 FFN 前一次。

---

## 二、LayerNorm 三步拆解

2016 年的经典答案。校准一个向量，分三步：

```
八维向量 [x₁, x₂, ..., x₈]
    ↓
Step 1: 算均值 μ，每个数减掉 μ → 整体平移归零
    ↓
Step 2: 算标准差 σ，全体除一遍 → 波动压进 [-1, 1] 范围
    ↓
Step 3: 乘可学习参数 γ（伽马），加可学习参数 β（贝塔）
        → 让模型自己决定往回拉多少
```

| 步骤 | 操作 | 作用 |
|------|------|------|
| **1. 减均值** | `x - μ` | 平移归零 |
| **2. 除标准差** | `(x - μ) / σ` | 缩放不变性，压进 ±1 |
| **3. γβ 仿射** | `γ × x + β` | 恢复表达能力 |

> [!note] 参数成本
> γ 和 β 是**两条完整的参数向量**。这套操作又管零点，又管音量，非常周到。

---

## 三、RMSNorm：三步变一步

2019 年有人发出灵魂疑问：**第一步和第二步真的有必要吗？**

RMSNorm 的答案简单粗暴：

- 不算均值
- 不减均值
- 不要 β

同一个向量，这次只算一个数——**RMS（Root Mean Square，均方根）**：

```
RMS = √(mean(x²))
```

物理意义就是这组数的**平均能量**。

```
所有维度一起除以 RMS
    ↓
条形形状完全没变，只是等比例缩小
    ↓
能量归一，均值信息保留
    ↓
乘个 γ 收工
```

> [!important] 关键区别
> RMSNorm 保留了原始的均值信息，**只把音量拧到标准刻度**。
>
> 三步变一步，两组参数变一组。

### LayerNorm vs RMSNorm 对比

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| **减均值** | 有 | 无 |
| **除标准差** | 有 | 有（换为 RMS） |
| **γ 参数** | 有 | 有 |
| **β 参数** | 有 | **无** |
| **参数数量** | 2 条向量 | **1 条向量** |
| **计算步骤** | 3 步 | **1 步** |
| **分布形状** | 被改变 | **保留原始分布** |

---

## 四、凭什么效果不掉？

### 拆解实验结论

> [!abstract] Zhang & Sennrich 2019
> LayerNorm 的功劳几乎全部来自**"缩放不变性"**（把幅度拧回标准这件事），而**"平移归零"**那一步对效果的贡献微乎其微。

> [!tip] 一句话总结
> **减均值是装饰，除幅度才是承重墙。**

### 拆掉装饰，白赚四样东西

| 收益 | 说明 |
|------|------|
| **参数减半** | 每个 Norm 层少一整条 β 向量 |
| **计算更省** | 不用先算均值再算方差，一次平方和搞定 |
| **梯度更稳** | 除数是能量 + ε，不容易趋近零，深层网络不怕梯度消失 |
| **效果一分不掉** | 核心能力（缩放不变性）完全保留 |

### 硬数据

| 指标 | LayerNorm | RMSNorm | 变化 |
|------|-----------|---------|------|
| **训练速度**（1000 步） | 665s | 501s | **快了 24.7%** |
| **BLEU**（翻译质量） | 23.6 | 23.7 | **+0.1，不掉反升** |

### Google 大规模复现证据

> [!important] 《Do Transformer Modifications Transfer?》
> Google 把几十种 Transformer 魔改统统重跑一遍，绝大多数换个框架就失灵，而 **RMSNorm 是极少数真正有效的架构魔改**。
>
> **SuperGLUE：71.66 → 75.45**，训练速度还更快。

**便宜、更快、不掉分。** LLaMA 之后没人再回头。

---

## 五、手撕代码对比

### LayerNorm

```python
# 两条参数向量：gamma, beta
# 前向三步：算均值 → 算方差 → 减完除完再仿射变换
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
x = (x - mean) / torch.sqrt(var + eps)
x = gamma * x + beta
```

### RMSNorm（LLaMA 同款）

```python
# 只有一条参数向量：gamma
# 前向就一行：x 平方取均值 → 加 eps 开根号 → 取倒数 → 直接乘回去
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
x = gamma * x / rms
```

> 就这么两行的差别：参数少一半，归约少一趟。

### PyTorch 官方实现

```python
# PyTorch 已收录官方标准库
norm = nn.RMSNorm(hidden_size, eps=1e-6)
```

---

## 六、行业现状

今天清一色 RMSNorm 的模型：

| 模型 | 归一化方式 |
|------|-----------|
| **LLaMA** | RMSNorm |
| **Qwen** | RMSNorm |
| **Mistral** | RMSNorm |
| **DeepSeek** | RMSNorm |
| **Gemma** | RMSNorm |

---

## 关键 Takeaways

| # | 要点 |
|---|------|
| **1** | 大模型几十层堆叠数值会指数失控，必须在每层入口归一化（Pre-Norm） |
| **2** | LayerNorm 三步：归零（减均值）→ 压幅（除标准差）→ 仿射（γβ 拉回），两组参数 |
| **3** | RMSNorm 只留一步：除以均方根 RMS（平均能量），只管音量不管零点，1 组参数 |
| **4** | 拆解实验证明：LayerNorm 的功劳几乎全来自"缩放不变性"，"平移归零"贡献微乎其微 |
| **5** | RMSNorm 提速 24.7%，参数减半，效果不掉，Google 复现盖章（SuperGLUE +3.79） |
| **6** | LLaMA/Qwen/Mistral/DeepSeek/Gemma 全员标配，PyTorch 已收录 `nn.RMSNorm` |

> [!quote] 一句话带走
> LayerNorm 又管零点又管音量，RMSNorm 发现**零点根本没人在乎**——少做一半的事，一分都不丢。
