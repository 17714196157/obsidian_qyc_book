---
title: "TP并发数值不稳"
source: "https://www.kimi.com/chat/19ef3f12-44a2-8845-8000-09faff94cb69?chat_enter_method=history"
author:
published:
created: 2026-06-23
description: "Kimi K2.6 模型发布！全新建站功能可生成极具设计感的网站，支持轻量后端模块；Agent 集群全面升级，Office 文档一键转可复用技能。Claw 群组同步开启内测，探索多 Agent 协作的全新可能。"
tags:
  - "clippings"
---
nohup vllm serve /home/model/Qwen25-14B-Instruct/ --tensor-parallel-size 2 --trust-remote-code --max-model-len 32768 --gpu-memory-utilization 0.95 --host 0.0.0.0 --port 8103 --served-model-name qwen2.5-14b-instruct > /var/log/qwen2.5-14b-instruct.log 2>&1 & 这个命令下vllm请求在几百并发下，返回结果异常乱说，但是去掉--tensor-parallel-size 2 --trust-remote-code 就好了，这是为什么？

## 核心原因分析

### 1\. Tensor Parallelism 的 All-Reduce 非确定性问题

vLLM 的 TP 实现基于 Megatron-LM 的张量并行算法，需要在多个 GPU 之间进行频繁的 **all-reduce 通信** 来聚合计算结果（如 Row Parallelism 后的结果求和）。

**高并发下的问题：**

- **浮点运算的非结合性** ： `(a + b) + c ≠ a + (b + c)` ，不同 GPU 的 all-reduce 求和顺序不一致，导致数值误差累积
- **并发请求交错执行** ：几百并发下，不同请求的 token 计算在 GPU 上交错执行，all-reduce 的时序和分组可能产生 race condition
- **KV Cache 分片不一致** ：TP 下 KV Cache 也分片存储，高并发时不同请求的 cache 访问模式可能导致同步问题

根据研究论文 ，这正是 **"TP-induced inconsistency"** 问题：即使 greedy decoding (`temperature=0`)，不同 TP size 或 batch size 也会产生不同输出，原因是 **"non-associativity of floating-point arithmetic and inconsistent reduction orders across GPUs"** 。

### 2\. Multiprocessing (mp) Executor 的已知 Bug

vLLM 单节点默认使用 Python `multiprocessing` 作为分布式执行后端。搜索结果显示 ， **mp backend 在 v0.12.0 等版本存在严重的输出损坏 bug** ：

```markdown
# BROKEN: default mp backend
vllm serve Qwen/Qwen3-VL-2B-Instruct --tensor-parallel-size 4

# FIXED: ray backend
vllm serve Qwen/Qwen3-VL-2B-Instruct --tensor-parallel-size 4 --distributed-executor-backend ray
```

**关键发现** ：在 Qwen3-VL 的测试中，mp backend 导致准确率从 52% 暴跌到 12.3%，而 **切换到 `ray` backend 完全修复** 。

### 3\. --trust-remote-code 的潜在影响

虽然日志显示 `trust_remote_code` 对 vLLM serve "has no effect here and is ignored" ，但如果你的模型需要自定义架构代码：

- **代码质量** ：远程代码可能包含非线程安全的操作，在 TP + 高并发下暴露 race condition
- **初始化顺序** ：不同 worker 进程加载自定义代码的时序差异，可能导致模型状态不一致

#### 解决方案1：切换到 Ray Backend（推荐）

这是经过验证的最有效解决方案 ：

```bash
nohup vllm serve /home/model/Qwen25-14B-Instruct/ \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 \
  --port 8103 \
  --served-model-name qwen2.5-14b-instruct \
  --distributed-executor-backend ray \  # 关键：切换到 ray
  > /var/log/qwen2.5-14b-instruct.log 2>&1 &
```

#### 验证方法
测试 TP=2 下输出是否稳定：

```bash
# 发送相同请求多次，检查一致性
for i in {1..5}; do
  curl http://localhost:8103/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-14b-instruct",
      "messages": [{"role": "user", "content": "2+2=?"}],
      "temperature": 0
    }'
done
```

**如果添加 `--distributed-executor-backend ray` 后多次请求输出一致（都返回 "4"），则确认是 mp backend 的 bug。**根本原因是 vLLM 默认的 `multiprocessing` executor 在 Tensor Parallelism 场景下存在数值正确性 bug** ，高并发时 all-reduce 的浮点非确定性和进程间同步问题导致输出乱说。切换到 **`ray` backend** 是最可靠的解决方案，这也是 vLLM 社区针对此类问题的标准修复方式。

### 核心问题：浮点数加法的"顺序陷阱"

#### 1)浮点数不满足结合律

这是计算机科学的基础知识，但很多人没意识到它的严重性：

```python
# 在 Python 中验证
a = 1.0000001
b = 1.0000002  
c = -1.0000001

# 数学上：(a + b) + c == a + (b + c)
# 实际上：
print((a + b) + c)  # 结果：1.0000002
print(a + (b + c))  # 结果：1.0000001
# 两者不相等！
```

**为什么会这样？**

- 浮点数只有 32/64 位精度
- 大数吃小数： `1e8 + 1 = 1e8` （1 被"吃掉"了）
- 不同加法顺序导致不同的精度损失

#### 2\. TP 如何触发这个问题

##### 单 GPU（无 TP）的计算流程：

```markdown
输入向量 x → Linear 层计算 → 得到输出 y
```

简单直接， **顺序完全确定** 。

##### TP=2（双 GPU）的计算流程：

假设 Linear 层的权重矩阵被切成两半，分别放在 GPU0 和 GPU1：

```markdown
输入 x ──────────────────┐
                         │
    ┌────────────────────┴────────────────────┐
    │                                         │
    ▼                                         ▼
 GPU 0 计算: y0 = W0 @ x                   GPU 1 计算: y1 = W1 @ x
    │                                         │
    │         ┌───────────────────┐           │
    └────────►│   All-Reduce      │◄──────────┘
              │  (求和: y = y0+y1) │
              └───────────────────┘
                         │
                         ▼
                      最终输出 y
```

**关键步骤：All-Reduce 求和**

```python
# GPU0 和 GPU1 各自算出一部分结果后，需要汇总
# 假设：
GPU0 的结果片段: [1.0000001, 2.0000001]
GPU1 的结果片段: [1.0000002, 2.0000002]

# All-Reduce 求和：
最终结果 = [1.0000001 + 1.0000002, 2.0000001 + 2.0000002]
        = [2.0000003, 4.0000003]
```

看起来没问题？ **但问题出在并发时** 。

#### 3\. 高并发下的"乱序"问题

##### 场景：100 个请求同时进来

**单 GPU（无 TP）的情况：**

```markdown
请求1: x1 → 计算 → y1
请求2: x2 → 计算 → y2  
请求3: x3 → 计算 → y3
...
```
每个请求独立计算， **内部顺序固定** ，结果稳定。
**TP=2 的情况：**
```markdown
时间片 1:
  GPU0: [请求1的部分, 请求3的部分, 请求5的部分, ...]
  GPU1: [请求1的部分, 请求3的部分, 请求5的部分, ...]
  All-Reduce: 汇总请求1,3,5...

时间片 2:
  GPU0: [请求2的部分, 请求4的部分, 请求6的部分, ...]
  GPU1: [请求2的部分, 请求4的部分, 请求6的部分, ...]
  All-Reduce: 汇总请求2,4,6...
```
**问题出现了：**
```python
# 假设有 3 个请求的部分结果需要在 GPU 间汇总
# 请求 A: [1.0, 1e-8]      # 1e-8 是很小的数
# 请求 B: [1.0, 2e-8]
# 请求 C: [1.0, 3e-8]

# 情况 1：按 A→B→C 顺序求和
sum = 1.0 + 1e-8 + 1.0 + 2e-8 + 1.0 + 3e-8
    = 3.0 + 6e-8
    ≈ 3.00000006  (保留了小数)

# 情况 2：按 B→A→C 顺序求和（并发调度不同）
sum = 1.0 + 2e-8 + 1.0 + 1e-8 + 1.0 + 3e-8
    = 3.0 + 6e-8  
    ≈ 3.00000006  (理论上相同，但实际浮点运算可能不同)

# 更糟的情况：大数先加，小数被"吃掉"
# 如果某个中间结果变成了 10000.0，再加 1e-8：
10000.0 + 1e-8 = 10000.0  (1e-8 被完全吃掉！)
```

**数值误差如何放大：**
```python
# 正常情况（单 GPU）：
token "4" 的 logits: [100.0, 1.0, 0.5, ...]  # "4" 的分数遥遥领先
softmax 后概率: [0.999, 0.0001, 0.00005, ...]
→ 确定选 "4"

# TP=2 高并发（数值误差导致）：
token "4" 的 logits: [100.0 + 0.0001误差, 1.0 - 0.0001误差, ...]
                   = [100.0001, 0.9999, ...]
softmax 后概率: [0.9990001, 0.00010001, ...]
→ 还是选 "4"，没问题

# 但如果误差累积到关键位置：
# 假设第 10 步，原本 "北京" 和 "上海" 的分数很接近：
正常 logits: [10.0, 9.9999, ...]  # "北京" 略高
误差 logits: [10.0002, 9.9997, ...]  # 误差改变了排名！
→ 模型突然开始说 "上海" 而不是 "北京"

# 更糟的是：一旦选错 token，错误会滚雪球
# "上海" 后面跟着 "的东方明珠"，而 "北京" 后面跟着 "的天安门"
# 整个输出就完全跑偏了
```
可视化：误差如何变成"幻觉"
```markdown
用户问：中国的首都是哪里？

正常推理链：
"中国" → "的" → "首都" → "是" → "北京" → "。"
概率:  0.99   0.98   0.97   0.96   0.95   0.99

TP=2 高并发（数值误差导致）：
"中国" → "的" → "首都" → "是" → "上海" → "的" → "东方" → "明珠"
概率:  0.99   0.98   0.97   0.96   0.51← 这里！ 0.6   0.5   0.4
                                       ↑
                              原本"北京"0.95，但误差让它变成0.49
                              "上海"从0.02变成0.51，反超了！
```

#### 6\. 为什么去掉 --tensor-parallel-size 2 就正常

| 维度             | TP=2（双 GPU）             | TP=1（单 GPU）   |     |
| -------------- | ----------------------- | ------------- | --- |
| **All-Reduce** | 需要，引入求和顺序不确定性           | 不需要           |     |
| **浮点精度**       | 多卡间通信可能用 FP16/BF16，精度更低 | 单卡内可用 FP32 累加 |     |
| **并发调度**       | 多卡任务调度复杂，顺序不确定          | 单卡顺序执行，完全确定   |     |
| **内存访问**       | 跨卡 PCIe/NVLink 传输有延迟和缓冲 | 显存内直接访问       |     |
| **结果**         | **非确定性** ，高并发时误差累积      | **确定性** ，结果稳定 |     |

#### 7\. 为什么 Ray Backend 能修复

```python
# vLLM 的两种分布式执行器：

# 1. Multiprocessing (mp) - 默认，有问题
进程 0 (GPU0) ──┐
                ├──► Python multiprocessing 通信 ◄── 不稳定，有 race condition
进程 1 (GPU1) ──┘

# 2. Ray - 更稳定
Worker 0 (GPU0) ──┐
                  ├──► Ray 分布式框架 ◄── 工业级，通信顺序严格受控
Worker 1 (GPU1) ──┘
```

**Ray 的优势：**

- **任务队列严格排序** ：确保 all-reduce 按固定顺序执行
- **通信原语更可靠** ：避免了 mp 的进程间竞争
- **更好的同步机制** ：确保所有 worker 在同一 step 做 all-reduce

**一句话总结**
> **TP 把一个大矩阵拆到多个 GPU 上算，算完后要"对答案"（all-reduce 求和）。高并发时，多个请求的"对答案"顺序被打乱，浮点数加法的精度误差累积，导致模型在某一步选错 token，然后像滚雪球一样越错越远，最终输出完全乱说。单 GPU 没有"对答案"环节，所以稳定。**

想象你在做一道复杂的数学题：
- **单 GPU** ：你一个人做，按固定顺序计算，结果稳定
- **TP=2** ：你和同桌分工做，每步要交换中间结果相加。如果教室里 100 组人同时在喊数字，你们听到的顺序可能不一样，导致加错了一个小数点。后面所有步骤都基于这个错误，最终答案完全不对。
**Ray backend 就像给每组人发了对讲机，规定必须按编号顺序报数，避免了混乱。**
