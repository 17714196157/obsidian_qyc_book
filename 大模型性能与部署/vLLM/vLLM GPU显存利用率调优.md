---
title: "vLLM GPU显存利用率调优"
source: "https://mp.weixin.qq.com/s/XzkEJRfFaiJYF7yvotE0cw"
author:
published:
created: 2026-07-10
description:
tags:
  - clippings
  - vllm
  - gpu
  - 显存调优
  - OOM
  - KV-Cache
---

> [!tip] 读完这篇你能带走什么
> 一套 vLLM OOM 步进式排查框架 + 5 个参数的生产级配置值
>
> **适用场景：** 单卡/多卡 vLLM 推理部署，显存不够用或频繁 OOM
> **不适用：** 已有成熟压测体系的大规模集群

我们线上第一个 vLLM 推理服务上线那天，凌晨 2 点告警响了——GPU 显存 OOM，服务挂了。

不是因为模型太大。A100 80GB 跑 32B 模型，怎么算都够。但一看 nvidia-smi，显存利用率才 60%，就已经 OOM 了。

浪费 40% 显存还能 OOM——这是 vLLM 默认配置最大的陷阱。

---

## 一、根因：KV Cache 预分配机制在吃你 40% 显存

vLLM 启动时做的事不是"按需分配显存"，而是"先把 KV Cache 池子画好了再干活"。

具体来说，vLLM 在启动阶段会预分配 KV Cache 区域，大小由两个参数决定：

```
KV Cache 预分配 ≈ max_model_len
                × max_num_seqs
                × 每层 KV 大小
```

vLLM 默认值是什么？

- `max_model_len` = 模型原生最大上下文（如 131072）
- `max_num_seqs` = 256

拿 Qwen 32B 模型在 A100 80GB 上算一笔账：

```
KV Cache = 131072 × 256 × (hidden_size × num_layers × 2 × dtype)
         ≈ 131072 × 256 × ~0.5MB/token
         ≈ 16.7 GB
```

就 KV Cache 预分配已经吃掉 16.7GB。加上模型权重约 20GB（BF16 精度），启动时已占 36.7GB。A100 80GB 的 `gpu_memory_utilization=0.90` 给出 72GB 的可用池——看起来还有余量。但只要多来几个并发请求，激活值和临时 buffer 一上来，OOM 只是时间问题。

> [!danger] "显存利用率 60% 还 OOM"的真相
> vLLM 默认的 `max_num_seqs=256` 是为服务器集群设计的。单卡/小集群直接用，40% 显存被 KV Cache 预分配白占了，根本没留给实际推理。

---

## 二、5 个参数的生产级配置——按优先级排序

别一上来就改 `gpu_memory_utilization`。参数之间有依赖关系，调错了顺序等于白调。我们线上验证过的调整顺序和推荐值：

### 参数 1：max_num_seqs（优先级最高）

| 项目 | 值 |
|------|-----|
| 默认值 | 256 |
| 生产推荐 | 4（单卡）/ 16-32（多卡） |

这是影响最大的单一参数——直接决定 KV Cache 预分配大小。256 是给能承受几百并发的服务器集群用的。

我们线上测过：把 `max_num_seqs` 从 256 降到 4，KV Cache 预分配从 16.7GB 掉到 0.26GB。这个参数一调，显存压力直接降一个数量级。

单卡场景 4 就够日常推理，多卡并行设 16-32。别担心"并发不够"——大部分推理服务的瓶颈在模型推理速度，不在排队。

### 参数 2：max_model_len（按业务需求设）

| 项目 | 值 |
|------|-----|
| 默认值 | 模型原生（如 131072） |
| 生产推荐 | 按 95 分位 token 数 × 1.5 设定 |

大部分推理场景不需要 131072 token 的上下文。我们线上 90% 的请求上下文不超过 32000 token。设 131072 等于给那 10% 不到的请求预分配了全部显存。

判断方法：看业务日志里 95 分位的 prompt+output token 数，乘以 1.5 作为 `max_model_len`。不需要就别开大。

### 参数 3：gpu_memory_utilization（最后调）

| 项目 | 值 |
|------|-----|
| 默认值 | 0.90 |
| 生产推荐 | 0.90（24GB 卡）/ 0.93（80GB 卡） |

先调好 `max_num_seqs` 和 `max_model_len`，再调这个。这是"池子多大"的全局约束，不是"KV Cache 多大"的控制。

我们线上实测（A100 80GB）：

- 0.90 → 稳定，但吞吐量没吃满
- 0.93 → 稳定，吞吐量提升约 15%
- 0.95 → 启动时偶尔失败（CUDA graph 编译吃掉剩余显存）

> [!important] 甜点值是 0.92-0.93。别碰 0.95。

不同 GPU 显存的安全上限：

| GPU 显存 | 安全上限 | 说明 |
| --- | --- | --- |
| 24GB（RTX 4090） | ≤ 90% | 余量小，保守设 |
| 48GB（L40S） | ≤ 92% | ECC 稳定，可稍高 |
| 80GB（A100） | ≤ 94% | 高压可接受 |

### 参数 4：enforce_eager（稳定性开关）

| 项目 | 值 |
|------|-----|
| 默认值 | False（启用 CUDA graph） |
| 生产推荐 | True（开启） |

CUDA graph 预编译计算图让第一轮推理快 300ms，但编译时额外占用显存——在显存紧张场景下这就是最后一根稻草。

我们线上测过：`gpu_memory_utilization=0.93` 时，不开 `enforce_eager` 偶尔启动失败。开启后稳定运行，第一轮推理延迟多 300ms，后续推理速度一致。

> [!tip] 显存紧张就开，显存充裕可以不碰。

### 参数 5：enable_chunked_prefill（长上下文保险）

| 项目 | 值 |
|------|-----|
| 默认值 | False |
| 生产推荐 | `max_model_len ≥ 65536` 时开启 |

长 prompt 一次性加载会造成显存尖峰。`chunked_prefill` 把 prompt 切成小块逐批处理，显存曲线从"尖刺"变成"平坡"。

不是所有场景都需要——如果 `max_model_len` 已经设得较小（≤32000），这个参数收益不大。

---

## 三、OOM 排查三步法——比"挨个调参数"快 10 倍

不要上来就改参数。我们线上总结的排查顺序：

### 第一步：看日志，定位谁在吃显存

```bash
grep "GPU KV cache size" vllm.log
```

如果 KV Cache 超总显存 50% → 问题在 `max_num_seqs` 或 `max_model_len`。

### 第二步：降 max_num_seqs，不降 gpu_memory_utilization

大多数人的直觉是"OOM 了就降 `gpu_memory_utilization`"。这治标不治本——池子小了，KV Cache 占比还是高，只是推迟了 OOM 的时间。

正确做法：先降 `max_num_seqs`，直接压缩 KV Cache 预分配。

### 第三步：调 max_model_len 到业务实际需求

降了 `max_num_seqs` 还 OOM？说明 `max_model_len` 设太大了。砍到业务 95 分位 token 数 × 1.5。

### 真实案例

我们线上一个真实案例：Qwen 32B + A100 80GB，默认配置启动后 5 分钟 OOM。排查结果：

| 阶段 | 配置 | KV Cache | 占总显存 | 结果 |
|------|------|---------|---------|------|
| **根因** | `max_num_seqs=256` × `max_model_len=131072` | 16.7GB | 21% | 启动后 5 分钟 OOM |
| **修复** | `max_num_seqs=8`, `max_model_len=65536` | 2.1GB | 2.6% | - |
| **结果** | - | - | - | 显存利用率 60% OOM → 89% 稳定运行 2 周 |

---

## 四、你明天就能用的行动清单

### 命令 1：生产级启动配置（A100 80GB + 32B 模型）

```bash
vllm serve /path/to/model \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.93 \
  --enforce-eager \
  --enable-chunked-prefill
```

### 命令 2：单卡消费级 GPU 配置（RTX 4090 24GB + 7B）

```bash
vllm serve /path/to/model \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --dtype auto
```

### 上线前自检清单

- [ ] 启动后立刻查 "GPU KV cache size"，确认 KV Cache < 总显存 30%
- [ ] `nvidia-smi -l 1` 跑 5 分钟，显存占用不持续上升
- [ ] 发一条 `max_model_len` 长度的请求，确认不 OOM，不断流
- [ ] 连续发 10 条请求（间隔 1 秒），确认无超时、无 OOM

---

> [!quote] 总结
> 显存不是不够用——是被预分配机制白吃了。
