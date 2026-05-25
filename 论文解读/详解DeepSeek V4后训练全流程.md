---
title: "详解DeepSeek V4后训练全流程"
source: "https://mp.weixin.qq.com/s/ug9k45wZpweRt-XFuiKL1Q"
author:
  - "[[卖铁观音的小男孩]]"
published:
created: 2026-05-25
description: "DeepSeek V4后训练祭出全新范式：先用SFT+GRPO独立炼制各领域&quot;偏科专家&quot;，再通过全词表同策略蒸馏（OPD）将多个万亿参数专家无损融入同一模型，配合Actor-GRM一体化奖励与极致工程优化，实现&quot;先偏科、再合体、能力反增&quot;的跨越式突破。"
tags:
  - clippings
---

> [!abstract] 来源信息
> 原创 卖铁观音的小男孩 *2026年5月24日 22:13*

## 引言

半枕松风茶未熟，吟怀潇洒满腔春。小伙伴们好，我是微信公众号"小窗幽记机器学习"的小编卖铁观音的小男孩。本系列此前已详细介绍：[DeepSeek V4 模型架构](https://mp.weixin.qq.com/s?__biz=MzkwNjE2ODMxNQ==&mid=2247490869&idx=1&sn=eb9f92db2e1f3e0e71e811f139efc951&scene=21#wechat_redirect)、[DeepSeek V4中 Muon 优化器的前世今生](https://mp.weixin.qq.com/s?__biz=MzkwNjE2ODMxNQ==&mid=2247490874&idx=1&sn=b64a8107ef34f0dcc5eab05e62894163&scene=21#wechat_redirect)、[V3→V3.2→V4：DeepSeek预训练演变](https://mp.weixin.qq.com/s?__biz=MzkwNjE2ODMxNQ==&mid=2247490888&idx=1&sn=954d885c58fc29ee9f7a9ed68099c3a3&scene=21#wechat_redirect)、[深入理解DeepSeek V4 On-Policy Distillation背后的Reverse KL](https://mp.weixin.qq.com/s?__biz=MzkwNjE2ODMxNQ==&mid=2247490888&idx=2&sn=016e5672736658fd6d7164740e3fc5c3&scene=21#wechat_redirect)、[大模型后训练新范式：On-Policy Distillation (OPD)](https://mp.weixin.qq.com/s?__biz=MzkwNjE2ODMxNQ==&mid=2247490896&idx=1&sn=444677d43a23ba0f0ec218e49d130cc1&scene=21#wechat_redirect)。本文聚焦 DeepSeek V4 **后训练（Post-Training）Pipeline**。DeepSeek V4 的后训练没有沿用 V3.2 时期的混合强化学习路线，而是彻底采用了一套新打法：

> [!tip] 核心打法
> **先独立训练多个领域的"偏科专家"；再通过同策略蒸馏（OPD）熔于一炉，把所有能力整合进同一个模型——一拆一合，能力反而更强。**

下面我们按照"领域专家训练 → OPD 原理 → V4版OPD实现 → 工程基建"四步，逐一拆解 DeepSeek V4 的后训练流程。

*(顺便安利：如果你对大模型底层架构也充满求知欲，强推我朋友刚出的新书《Transformer 技术纵深:架构解析与前沿突破》。)*

---

## 1. 领域专家训练

第一阶段的核心目标是针对数学、代码、Agent、指令遵循等垂直领域，独立训练出一批极致的领域专家模型。具体做法是用高质量的垂直领域数据对基础模型（Base Model）进行 SFT（监督微调），随后进入 RL（强化学习）阶段。RL 算法依然使用 GRPO，通过特定领域的 Prompt 和奖励信号，持续逼近单一领域的能力天花板。

### 1.1 推理程度（Reasoning Efforts）

为了让模型具备灵活的思考深度，V4 在 RL 阶段通过分配不同的长度惩罚（Length Penalties）和上下文窗口，训练出了三种推理模式（通过 `<think>` 标签触发）：

| 模式 | 说明 |
|------|------|
| **Non-think** | 快速直觉响应，用于日常低风险任务 |
| **Think High** | 具备逻辑分析能力，适用于中等复杂度问题 |
| **Think Max** | 驱动模型探索推理极限 |

为了激发 **Think Max** 模式，会在 System Prompt 中嵌入以下指令：

```
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the
problem to resolve the root cause, rigorously stress-testing your logic against all potential
paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every intermediate
step, considered alternative, and rejected hypothesis to ensure absolutely no assumption
is left unchecked.
```

![[file-20260525161928898.png]]

推理深度解决的是"模型如何思考"的问题；而如何判断这些思考是否正确、质量是否足够高，则是奖励模型需要回答的核心问题。

### 1.2 DeepSeek 奖励模型的演进

如何评估那些难以验证的任务，一直是强化学习（RL）在复杂对齐场景中的核心难题。从 V3 到 V4，DeepSeek 在奖励模型（Reward Model, RM）的设计上经历了一次系统性演进：彻底抛弃了传统的黑盒标量奖励模型，走向了「生成-评判」同源的联合优化。

#### 演进路线：V3 → V3.2 → V4

用一句话总结这一演进：从"引入思维链的独立裁判"，到"持标准量表的精细裁判"，最终进化为"与运动员共同成长的自我进化型裁判"。

**DeepSeek V3【开创范式：CoT-based GRM】**

- **核心痛点**：传统标量 RM 容易被 Reward Hacking（奖励作弊），且是个不可解释的"黑盒"。
- **解决方案**：首次提出**生成式奖励模型（GRM）**。用 SFT 检查点训练一个语言模型，强制其在给出奖励分数前先输出**思维链（CoT）**。
- **定位**：GRM 与 Actor（策略网络）是**分离**的。GRM 充当一个具备推理能力的独立考官。

**DeepSeek V3.2【尺度升级：Rubric-guided GRM】**

- **核心痛点**：开放式通用任务（如 Agent、长文本）缺乏统一的评判尺度。
- **解决方案**：引入**Prompt-specific Rubrics（逐提示词专属量表）**。为每个 Prompt 定制详细的评估维度，GRM 根据这些 Rubrics 进行打分。
- **定位**：GRM 的评判变得极度精细化和标准化，但它依然是为主网络 RL 提供信号的"外部工具"。

**DeepSeek V4【终极合并：Actor-GRM Joint Optimization】**

- **核心痛点**：外部裁判的能力上限会锁死 Actor 的上限（即 RM 无法评估比自己更聪明的 Actor）。
- **解决方案**：**Actor 网络直接作为 GRM（Actor=GRM）**。在 RL 阶段，直接对 GRM 的评判能力本身进行强化学习优化。
- **定位**：网络既是生成者，又是评判者。评判（Judging）与生成（Generation）能力在同一个网络、同一个 RL 过程中被联合优化。

#### DeepSeek V4 GRM 解析

V4 技术报告中对 GRM 的描述虽然简短，但蕴含了极高的信息量。其底层逻辑是：**"评判别人的答案好坏"本身也是一种极其复杂的推理任务**。V4 的 GRM 机制设计本质上是完成了以下三大突破：

> [!important] 突破一：架构解构——消灭独立 RM，实现「同体优化」
> 在 V4 中，不存在独立的奖励模型。模型在 RL 过程中同时扮演两个角色：一方面生成策略轨迹，另一方面通过读取衡量表（Rubrics）对轨迹进行评价。由于 RL 优化直接作用于这个统一体，模型在"学习如何解决问题"的同时，也在"学习如何鉴别好答案"。**这种联合优化让模型的内在推理能力自然地转化为评估能力，反之亦然。**

> [!tip] 突破二：数据杠杆——极少量标注撬动强泛化评判
> 传统 RLHF 需要海量（十万/百万级）的人类偏好对来训练 RM。而 V4 通过融合模型的自身逻辑，仅需要**极少量的多样化高质量人类标注数据**。因为一旦模型理解了底层逻辑，它就能通过自身的泛化能力，利用少量的 Rubric 指导，自动在复杂任务中推导出合理的评分。

> [!success] 突破三：根治作弊——逻辑同源防御 Reward Hacking
> 当评估者（传统标量 RM）缺乏真正的逻辑理解能力时，生成者（Actor）就会通过寻找捷径来骗取高分。在 V4 的 Actor-GRM 一体化架构下，由于"打分"是通过模型自身最强（Max 级别）的显式推理过程（思考标签 `<think>...</think>` 内的推演）得出的，**捷径被逻辑的高墙堵死。评判的鲁棒性（Robust scoring）达到了前所未有的高度。**

V4 的这一设计真正触及了自我对齐（Self-Alignment）与自我奖励（Self-Rewarding）的门槛——不再用一个"更差"的模型去指导一个"更好"的模型，生成与评估在同一网络中联合优化，使模型在开放域深层推理任务中实现持续的性能 Scaling。

### 1.3 专为长文本设计的 Agent 与工具调用机制

解决了推理深度与评估机制之后，领域专家训练的另一个重要维度，是如何让模型在真实的 Agent 任务中高效调用工具。围绕 1M Token 的超长上下文，专家训练在工具调用（Tool-Calling）上做了三项改良：

**1、全新 XML Schema**

引入特殊 Token `<|DSML|>` 并改用 XML 格式，大幅降低了工具调用时的格式解析错误率。

```
## Tools
You have access to a set of tools to help answer the user's question. You can
invoke tools by writing a "<|DSML|tool_calls>" block like the following:
<|DSML|tool_calls>
<|DSML|invoke name="$TOOL_NAME">
<|DSML|parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE
</|DSML|parameter>
...
</|DSML|invoke>
<|DSML|invoke name="$TOOL_NAME2">
...
</|DSML|invoke>
</|DSML|tool_calls>
String parameters should be specified as is and set 'string="true"'. For all
other types (numbers, booleans, arrays, objects), pass the value in JSON
format and set 'string="false"'.
If thinking_mode is enabled (triggered by <think>), you MUST output your
complete reasoning inside <think>...</think> BEFORE any tool calls or
final response.
Otherwise, output directly after </think> with tool calls or final response.
### Available Tool Schemas
{Tool Definition...}
You MUST strictly follow the above definedtool name and parameter schemas to
invoke tool calls.
```

**2、交错思考（Interleaved Thinking）**

在 V3.2 中，模型每轮回复结束后都会清空思考轨迹。而 V4 在 Agent/工具调用场景下，会**完整保留跨越多轮 User 消息的所有思考轨迹（Thinking Traces）**，确保在长线任务中维持连贯的思维链；在普通闲聊场景中则继续采用清空策略以节省 Token。

**3、快捷指令（Quick Instruction）**

针对判定搜索意图、信息源权威度等"前置辅助任务"，V4 直接在输入序列末尾追加特殊的快捷指令 Token（如 `<|query|>`、`<|domain|>`），通过复用当前请求的 KV Cache 实现**并行推理**。这彻底消除了重复 Prefill 的计算冗余，并极大降低了首字延迟（TTFT）。触发哪些快捷指令，由系统后台依据轻量级规则与对话状态动态决定。

![[file-20260525161928896.png]]

为了更直观地理解，下面举例说明快捷指令的运转方式。

**用户输入（Prompt）：**

> *"微信公众号<小窗幽记机器学习>最近文章的作者是谁？Ta卖的是什么？"*

**Step 1：用户输入与 KV Cache 复用**  
基础前向传播，KV Cache 存入显存，后续所有快捷指令全部复用。

**Step 2：快捷指令并行触发（毫秒级瞬发）**  
系统在 Prompt 后同时挂载多个快捷指令 Token 并行预测：

- **触发 `<|action|>`**：
  - 模型分析：涉及"最近文章"，属于时效性强的信息，无法单靠内部权重回答。
  - **快速输出：需要触发联网搜索。**
- **触发 `<|query|>`**：
  - 模型基于上下文，直接提取最高效的搜索词。
  - **快速输出：**`["小窗幽记机器学习 最新文章 作者", "小窗幽记机器学习 笔名 卖什么"]`
- **触发 `<|domain|>` 与 `<|authority|>`**：
  - 模型判断这是技术自媒体/社交媒体领域的查询，对信息权威性要求中等（搜微信公众号文章即可）。

**Step 3：外围系统执行搜索与主模型作答**

- 搜索引擎基于 `<|query|>` 的结果抓取网页资料。
- 抓取到的资料返回给主模型（资料中显示该公众号作者笔名常为"卖XXX的小男孩/小女孩"）。
- 主模型整合检索结果，向用户流式输出回答：

> *"微信公众号<小窗幽记机器学习>近期文章的作者笔名通常是「卖XXX的小男孩/小女孩」（如卖火柴的小男孩等，根据具体文章可能有所变化）。"*

**Step 4：对话结束，生成标题**

- 回答生成完毕后，系统静默触发 `<|title|>` 指令，复用全程上下文。
- 快速输出对话框侧边栏标题：`小窗幽记机器学习作者及笔名解析`

至此，领域专家训练阶段在推理模式、奖励机制、工具调用三个维度完成了全面升级。然而，训练出十几个能力各异的领域专家，只是成功了一半——更大的挑战在于：如何将这些专家的能力无损地融合进同一个模型，而不让它们相互干扰、顾此失彼？要理解 DeepSeek V4 给出的方案，需要先从 [OPD](https://mp.weixin.qq.com/s?__biz=MzkwNjE2ODMxNQ==&mid=2247490896&idx=1&sn=444677d43a23ba0f0ec218e49d130cc1&scene=21#wechat_redirect) 的设计逻辑说起。

---

## 2. 同策略蒸馏（On-Policy Distillation, OPD）

### 2.1 两种后训练范式的困境

LLM 后训练主要分为两类方法：

- **On-policy 训练（强化学习）**：从模型自身采样轨迹，依据结果奖励更新参数。优点是训练样本来自模型自身分布，泛化性强；缺点是**奖励极度稀疏**——每条轨迹只产生一个序列级信号，模型无法得知错误究竟出在哪一步，导致学习效率低下。
- **Off-policy 训练（SFT/蒸馏）**：让 Student 模仿 Teacher 生成的示例数据。优点是每个 Token 都提供梯度信号，奖励稠密；缺点是 Student 学的是**Teacher 常见的上下文分布**，而非自己实际推理时遇到的分布。一旦 Student 早早走入 Teacher 从未犯过的错误，便会持续偏离训练分布，误差随序列长度加速累积——在长文本场景下尤为致命。

两种方法各有长短，如下表所示：

| 方法 | 采样来源 | 奖励信号 |
|------|----------|----------|
| SFT（Off-policy 蒸馏） | Off-policy | 稠密 |
| 强化学习 | On-policy | 稀疏 |
| **On-policy 蒸馏** | **On-policy** | **稠密** |

### 2.2 OPD：兼得两者之长

On-policy 蒸馏（OPD）的核心思想正是打破上述二选一的困境：**从 Student 自身采样轨迹（保持 On-policy），同时让 Teacher 对轨迹中的每个 Token 给出评分（保持稠密监督）**。

具体而言，OPD 以逐 Token 的作为损失函数：

这意味着 Teacher 会对 Student 生成的每一步——包括导致最终错误的关键"分叉点"——进行精准惩罚，而不是仅在序列末尾给出一个粗粒度的对/错判断。

实验数据印证了这一优势：据 Qwen3 技术报告，使用 OPD 将 AIME'24 基准从 60% 提升至 74.4%，所需算力仅为等效 RL 训练的约**1/10**。

![[file-20260525161928876.png]]

理解了 OPD 的设计逻辑，一个现实问题随之而来：当 Teacher 不是一个，而是十几个千亿/万亿参数的领域专家时，V4 是如何把 OPD 推向极致的？

---

## 3. DeepSeek V4 版 OPD

DeepSeek V4 的答案是：**全词表多教师同策略蒸馏**。相比 V3.2 时代的混合数据 RL，这套方案从根本上规避了灾难性遗忘与任务间干扰，具体体现在以下三个设计层面。

**1、动态上下文匹配学习**

在 OPD 阶段，统一模型（Student）自己生成输出轨迹（保持 On-policy），并同时参考个专家模型（Teachers）的输出分布。底层逻辑确保 Student 会**根据当前任务的上下文，选择性地向最相关的专家对齐**（例如做数学题时，自动对齐数学专家的 Logits 分布）。

**2、逆 KL 散度与 Logits 级对齐**

OPD 优化的核心是 Student 分布与各 Teacher 分布之间的。通过在 Logits 层面进行微观对齐，V4 将物理上独立的十几个专家权重中蕴含的知识，完整融合到了一个统一的参数空间中，彻底规避了传统权重融合导致的性能衰减。

**3、全词表 Logits 蒸馏（Full-Vocabulary Logit Distillation）**

这是 V4 提升训练稳定性的关键设计。以往的 OPD 为节省算力，通常只计算单 Token 级别的 KL 估计来替代真实策略损失，这会带来极大的梯度方差，往往引发训练不稳定甚至崩溃。V4 保留了完整的 128K 词表 Logits 分布来计算 KL 损失，换来了极其稳定的梯度估计和对 Teacher 知识的高保真还原。

然而，全词表 OPD 的设计虽然在理论上高效，在工程层面却面临极高的显存与算力压力——128K 词表、十几个万亿参数 Teacher、百万 Token 长文本，三者叠加之下，系统若无专项优化便将不堪重负。这正是 V4 在基础设施层面需要攻克的核心难题。

---

## 4. RL 与 OPD 基础设施

面对上述压力，DeepSeek 工程团队在底层基础设施上实施了一系列精细化优化，覆盖显存管理、调度策略与算子实现三个层面。

### 4.1 核心解密：全词表 OPD 的极致工程实现

支持无限数量的万亿参数 Teacher 模型进行 128K 全词表蒸馏，若直接显式化完整 Logits，无论写入显存还是磁盘，都将造成严重的内存溢出与 I/O 瓶颈。DeepSeek 团队通过以下几个精妙设计彻底突破了这一瓶颈：

| 优化手段 | 说明 |
|----------|------|
| **隐藏状态缓存** | 前向传播时**不显式计算 128K 的完整 Logits**，而是计算到倒数第二层，将输出的**低维隐状态（Hidden States）**存入缓存——其体积比完整 Logits 小了几个数量级 |
| **即时重构** | 计算 Loss 时，将低维隐状态取出，通过 Prediction Head 当场还原成 128K 的 Logits，算完立即丢弃 |
| **按需加载 Teacher（ZeRO-like Sharding）** | Teacher 的权重采用类似 ZeRO 的分片策略按需拉取，不长期占用计算显存，且全异步加载不阻塞关键路径 |
| **教师调度系统** | 将训练数据按所属 Teacher 重新全局排序，确保任意时刻 GPU 显存中至多加载一个 Teacher 的 Prediction Head |
| **自研底层算子** | 使用专属语言 `TileLang` 编写融合算子，使 KL 散度计算在寄存器层面完成，从根本上消除了冗余的动态内存分配 |

### 4.2 其他核心基建优化

除了为 OPD 量身定制的调度方案，V4 的基建还在以下维度为大规模后训练保驾护航：

> [!note] FP4 量化无缝整合
> 在 RL 采样（Rollout）和 Teacher 推理阶段，直接启用原生 FP4（MXFP4）量化加速并节省显存；在反向传播计算梯度时，利用 FP8 更大动态范围的特性，将 FP4 无损反量化回 FP8，全程无需修改复杂的后向训练代码。

> [!note] 抗中断的 Rollout 服务（WAL 机制）
> 引入 Token 级别的预写日志（Write-Ahead Log, WAL）。当生成任务被集群抢占时，系统持久化已生成的 Token；恢复时通过 WAL 重建 KV Cache 继续解码，不仅避免了算力浪费，更从机制上消除了因抢占偏向短序列而引入的长度偏差（Length Bias）。

> [!note] 百万 Token 调度与 DSec 弹性沙盒
> 针对 1M 长文本，将 Rollout 数据解耦为"轻量元数据"和"重度 Token 字段"，通过共享内存按需加载消除节点内冗余；同时构建了基于 3FS 存储的 DSec 弹性沙盒平台，底层抽象出 Function Call 到全量虚拟机（fullVM）的四种执行介质，支撑了数十万并发的 Agent 与代码安全评测。

正是这一系列精细化的工程设计，使得 V4 的 OPD 不只是理论上的优化方向，而是可以真正落地、稳定运行于超大规模训练任务中的工程现实。

---

## 5. 总结

> [!quote] 核心方法论
> DeepSeek V4 的后训练可概括为两步走：
> 1. **"SFT + GRPO 打造极致领域专家"**
> 2. **"全词表 OPD 实现无损合体"**
>
> 配合针对长文本特化的交错思考与快捷指令机制，以及精细化的显存与计算基础设施，V4 最终在当前开源模型中确立了显著优势。

这套方法论的深层逻辑在于：**先让模型"偏科"到极致，再让它"博采众长"**——这或许也为后续更大规模、更多领域的模型融合提供了一条可复现的路径。
