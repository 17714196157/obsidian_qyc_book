---
title: "vLLM吞吐量调优 - Kimi"
source: "https://www.kimi.com/chat/19d14c04-4202-86fe-8000-09fad39e4093?chat_enter_method=history"
author:
published:
tags:
  - "clippings"
---
vllm启动时有设置并发数据，和最大长度，但是我记录理论上按这两个参数计算kv缓存占用，服务器的显存更本就不够。为什么服务能正常启动。

## 核心原因：vLLM的动态内存管理机制

vLLM并非在启动时预分配所有可能的KV缓存，而是采用 **按需分配 + 分页管理** 的策略，这是其能正常启动的关键。

### 1\. PagedAttention：按需分页分配

vLLM的核心创新是 **PagedAttention** 机制：

- **不预分配连续内存** ：传统方法会为每个序列预分配 `max_seq_len` 的连续KV缓存，但vLLM将KV缓存划分为 **固定大小的块（blocks）**
- **动态页表管理** ：通过页表将逻辑块映射到物理块，只有在实际生成token时才分配物理内存块
- **内存池机制** ：启动时只预分配一个 **内存池** （默认占GPU显存的90%左右），但其中的块是未分配状态

### 2\. 启动参数的真实含义

你设置的参数（如 `max_num_seqs` 和 `max_model_len` ）是 **上限值** 而非实际分配值：

- `max_num_seqs` ：最大并发序列数，但 **不是同时活跃的序列都会达到最大长度**
- `max_model_len` ：单个序列的最大长度限制，但 **实际序列通常远小于此值**

### 3\. 动态调度与内存复用

vLLM通过以下机制动态管理内存：

- **Continuous Batching（连续批处理）** ：请求完成后立即释放KV块给新请求，不需要等待整个batch完成
- **Chunked Prefill** ：长提示被分块处理，避免一次性占用大量KV缓存
- **内存回收** ：序列结束后立即回收其KV块，而非保留到batch结束

### 4\. 实际内存计算公式

实际KV缓存占用取决于：

```markdown
实际占用 = 当前活跃序列数 × 各序列实际长度 × 2(K+V) × 层数 × 隐藏维度 × 精度字节数
```

而非理论最大值：

```markdown
理论最大值 = max_num_seqs × max_model_len × 2 × 层数 × 隐藏维度 × 精度字节数
```

### 5\. 内存碎片化处理

PagedAttention通过 **非连续存储** 和 **块级管理** ，将内存碎片从35%降低到5%，使得内存池中的空间可以被高效复用。

## 总结

**为什么服务能正常启动？**

1. **启动时只分配内存池，不分配具体KV缓存**
2. **KV缓存在推理过程中按需分配**
3. **短序列和长序列混合时，平均长度远小于max\_model\_len**
4. **完成的请求立即释放内存供新请求使用**

**你的理论计算为何不准？**

- 你计算的是 **最坏情况下的静态分配**
- 实际运行是 **动态分配 + 统计复用** ，真实占用通常只有理论值的10-30%

**建议** ：

- 关注 `gpu_memory_utilization` 参数（默认0.9），它控制预分配的内存池大小
- 监控实际运行时的 `vllm_cache_usage` 指标，而非理论计算
- 如果出现OOM，通常是在 **运行高峰期** 而非启动时，此时需要调整 `max_num_seqs` 或启用KV缓存卸载

---

vllm部署时最大长度和并发数，这两次参数我应该怎么设置才能保证我吞吐量最大喃

这是一个关于vLLM参数调优的实践问题。让我搜索一下vLLM吞吐量优化的最佳实践，特别是关于max\_model\_len和max\_num\_seqs的设置策略。

搜索

vLLM max\_model\_len max\_num\_seqs throughput optimization best practice

118 个结果

根据搜索结果，我来为你详细解答如何设置 `max_model_len` （最大长度）和 `max_num_seqs` （并发数）以实现 **吞吐量最大化** 。

## 核心原则：吞吐量的本质是GPU利用率 × 批处理效率

vLLM的吞吐量优化不是简单地把两个参数调到最大，而是要找到 **显存、计算、延迟** 的最佳平衡点。

---

## 一、关键参数联动关系

```markdown
吞吐量 ∝ (batch_size × avg_sequence_length) / (prefill_time + decode_time × steps)
```

**三个核心参数的制约关系** ：

| 参数 | 作用 | 调大影响 | 调小影响 |
| --- | --- | --- | --- |
| `max_model_len` | 单请求最大token数 | 支持更长上下文，但KV缓存块数线性增加 | 可能截断长请求，但显存更省 |
| `max_num_seqs` | 最大并发序列数 | 提高batch size，但会增加调度开销 | 降低并发，但减少抢占 |
| `max_num_batched_tokens` | 每批次最大token数 | 提高prefill效率，但增加单次计算量 | 降低batch效率，但延迟更稳定 |

---

## 二、实战调优四步法（推荐流程）

### 第一步：建立基线（用真实流量说话）

先用默认参数启动，收集24小时数据：

- **请求长度分布** ：统计80%请求的 `prompt_len` 和 `completion_len` 落在哪个区间
- **显存占用曲线** ：观察峰值是否接近90%
- **延迟分布** ：关注P50/P95/P99，特别是P99是否远高于P50（存在长尾阻塞）

> **案例** ：某教育场景发现85%请求集中在5120-18432 token，但默认 `max_model_len=4096` 导致40%请求被截断重试。

### 第二步：定向突破（单变量优先）

**先调 `max_model_len`** ：

- 从基线值开始，每次增加4096（如4096→8192→12288）
- **观察指标** ：
	- P99延迟是否显著下降（截断减少）
	- 显存峰值是否突破92%（OOM风险）
	- `vllm:gpu_cache_usage_perc` 是否持续>95%（缓存瓶颈）

> **最佳实践** ：在12288处达到平衡，P99下降37%，显存峰值稳定在86%。

**再调 `max_num_seqs`** ：

- 固定 `max_model_len` ，从128开始，以32为步长增减
- **观察指标** ：
	- 吞吐量（req/s）是否线性增长
	- P50延迟是否随并发上升而恶化
	- GPU利用率是否持续>70%

> **拐点发现** ：192是拐点——超过此值，吞吐量增长趋缓，P50延迟开始爬升。

### 第三步：压力验证（模拟最坏场景）

构建两类压测：

1. **长请求风暴** ：100%请求 `prompt_len=8192, max_tokens=8192` ，持续5分钟
2. **混合负载** ：70%短请求 + 30%长请求，模拟真实业务

**通过标准** ：

- 长请求风暴下P95<8s
- 混合负载下短请求不被"饿死"（P95<2s）
- 无OOM且 `num_preemptions_total` 接近0（无抢占）

### 第四步：上线监控（持续调优）

部署监控告警：

- **自动告警** ： `vllm:gpu_cache_usage_ratio > 0.88` 连续5分钟触发
- **请求画像** ：每日聚合长度分布，若80%分位数连续3天上移，启动复审
- **降级预案** ：预置 `prod-high` 和 `prod-safe` 两套配置

---

## 三、关键参数设置策略

### 1\. max\_model\_len：匹配业务实际，而非模型上限

**常见误区** ：

- ❌ "越大越聪明"：强行拉长到32768，但注意力在>16384位置已衰减
- ❌ "一刀切迁移"：把其他模型的8192直接套用，忽视模型DNA差异

**正确做法** ：

```bash
# 统计80%请求覆盖的长度区间
# 若80%请求 < 8192，则设置：
--max-model-len 8192  # 而非模型支持的32768
```

**收益** ：显存节省30-50%，减少无效计算。

### 2\. max\_num\_seqs：寻找吞吐量拐点

**设置公式** ：

```markdown
max_num_seqs ≈ (gpu_memory_utilization × 总显存 - 模型权重) / (avg_seq_len × kv_cache_per_token)
```

**实战建议** ：

- **开发调试** ：64（低显存安全）
- **主力生产** ：128-192（4090/6000Ada推荐）
- **极限压测** ：1（单请求极致性能）

**关键指标** ：

- 当 `num_preemptions_total` （抢占次数）>5/1000请求时，立即降低 `max_num_seqs`

### 3\. 联动参数：max\_num\_batched\_tokens

**作用** ：控制prefill阶段的batch size，对长文本模型至关重要。

**推荐配置** ：

```bash
# 开发调试（3090/4080）
--max-num-batched-tokens 8192 \
--max-num-seqs 64

# 主力生产（4090/6000Ada）
--max-num-batched-tokens 16384 \
--max-num-seqs 128

# 极限长文本（A100/H100）
--max-num-batched-tokens 32768 \
--max-num-seqs 1  # 专注单任务质量
```

### 4\. 显存利用率：gpu\_memory\_utilization

**硬件分档设置** ：

- **8GB卡** （3070/4070）：0.45
- **12GB卡** （3080/4080）：0.55
- **24GB+卡** （4090/A10）：0.6（VL模型封顶）

> **警告** ：VL模型（如Qwen3-VL）切勿超过0.6，视觉编码器临时缓冲区会导致OOM

---

## 四、吞吐量最大化的黄金组合

### 场景A：高吞吐通用服务（推荐）

```bash
vllm serve $MODEL \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \        # 覆盖80%请求
  --max-num-seqs 192 \          # 拐点测试后的最优值
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \     # 关键：避免长提示阻塞
  --dtype half
```

**适用** ：聊天机器人、通用API服务

### 场景B：长短混合负载

```bash
vllm serve $MODEL \
  --gpu-memory-utilization 0.92 \
  --max-model-len 12288 \       # 支持长CoT推理
  --max-num-seqs 160 \          # 略低于纯短文本场景
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --block-size 32               # 减少块数量，显存节省18%[^11^]
```

**适用** ：教育、财务分析等CoT场景

### 场景C：极致吞吐（牺牲首token延迟）

```bash
vllm serve $MODEL \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \        # 强制短文本，提高batch效率
  --max-num-seqs 256 \            # 最大化batch size
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --swap-space 8 \                # CPU offload兜底
  --scheduler-delay-factor 0.5    # 增加调度延迟，提高batch命中率
```

**适用** ：离线批处理、日志分析

---

## 五、避坑指南

| 陷阱 | 现象 | 对策 |
| --- | --- | --- |
| **"一刀切"参数迁移** | 直接套用其他模型的参数，导致截断或OOM | 始终基于业务长度分布调参 |
| **忽略 `block-size` 隐性影响** | 调高 `max_model_len` 后显存暴增但吞吐未提升 | 同步评估 `--block-size=32` （默认16），块数量减半 |
| **混淆并发数与并行度** | TP=4时 `max_num_seqs` 未调整，导致队列饱和 | `max_num_seqs` 应与 `tensor_parallel_size` 成正比（1.8-2.0倍） |
| **"显存够就往高设"** | VL模型设0.8，视觉编码器临时缓冲导致OOM | VL模型封顶0.6，保留≥20%余量 |
| **改完参数不验证** | 非法参数静默回退到默认值 | 必做三查：API确认、日志grep、实际请求测试 |

---

## 六、监控指标速查表

**关键指标** ：

- `vllm:gpu_cache_usage_perc` > 95% → 降低 `max_model_len` 或提高 `gpu_memory_utilization`
- `vllm:num_preemptions_total` 上升 → 降低 `max_num_seqs`
- `vllm:iteration_tokens_total` → 观察batch效率

**最终建议** ：

> 吞吐量优化不是一次性的，而是 **"基线→调优→压测→监控→再调优"** 的持续闭环。没有万能参数，只有最适合你业务流量特征的参数组合。

你说的真好，把上面的知识点正对如何确定vllm部署时参数的角度，整理一下形成一个程序员现场部署时的指南

编辑

复制

分享

好的，我来把上面的知识点整理成一份 **程序员现场部署时的实战指南** ，方便你快速查阅和按步骤执行。

---

## vLLM 现场部署参数速查指南

> **目标** ：10分钟内完成参数配置，确保吞吐量最大化且服务稳定

---

## 第一步：确认业务需求（2分钟）

### 1.1 收集关键数据

| 问题 | 获取方式 | 记录值 |
| --- | --- | --- |
| 80%请求的输入长度？ | 日志统计/产品经理 | `P80_prompt_len` = \_\_\_ |
| 80%请求的输出生成长度？ | 日志统计 | `P80_completion_len` = \_\_\_ |
| 最长可能请求？ | 业务方确认 | `Max_business_len` = \_\_\_ |
| 预期峰值QPS？ | 压测目标/业务方 | `Target_QPS` = \_\_\_ |
| GPU型号和显存？ | `nvidia-smi` | `GPU_Memory` = \_\_\_ GB |

### 1.2 快速决策树

```markdown
P80_prompt_len + P80_completion_len < 4096? 
  → 是: 走【短文本高吞吐模式】
  → 否: 走【长文本平衡模式】

Max_business_len > 16384?
  → 是: 必须开启 --enable-chunked-prefill
  → 否: 可选开启
```

---

## 第二步：核心参数计算（3分钟）

### 2.1 max\_model\_len 设置

```python
# 公式：覆盖95%业务场景，而非模型理论上限
max_model_len = min(
    Max_business_len * 1.2,           # 留20%余量
    模型支持的最大长度,                # 如32768
    16384 if 视觉模型 else 32768       # VL模型保守设置
)

# 快速参考：
# - 聊天机器人: 4096-8192
# - 代码生成: 8192-16384  
# - 长文档分析: 16384-32768（必须chunked-prefill）
```

### 2.2 max\_num\_seqs 计算

```python
# 估算公式（保守版）
available_memory = GPU_Memory * gpu_memory_utilization - 模型权重占用
kv_per_token = 2 * num_layers * hidden_size * dtype_size  # 约1-2MB/1k tokens
avg_seq_len = (P80_prompt_len + P80_completion_len) / 2

max_num_seqs = int(available_memory / (avg_seq_len * kv_per_token) * 0.8)  # 留20%余量

# 硬件快速参考：
# - 4090/24GB: 128-192
# - A100/40GB: 256-384  
# - A10/24GB: 96-128
# - 3090/24GB: 64-96
```

### 2.3 gpu\_memory\_utilization 设置

| GPU显存 | 非视觉模型 | 视觉模型(VL) |
| --- | --- | --- |
| 8GB | 0.45 | 不建议部署 |
| 12GB | 0.55 | 0.50 |
| 24GB | 0.90 | **0.60** |
| 40GB+ | 0.92 | 0.70 |

> ⚠️ **VL模型红线** ：无论显存多大， **不得超过0.60** ，视觉编码器临时缓冲会OOM

---

## 第三步：选择部署模板（2分钟）

### 模板A：短文本高吞吐（P80 < 4096）

```bash
vllm serve $MODEL \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --block-size 16 \
  --dtype half \
  --scheduler-delay-factor 0.5  # 牺牲一点延迟换batch效率
```

### 模板B：长文本平衡（P80 > 4096）

```bash
vllm serve $MODEL \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 12288 \       # 按业务实际设置
  --max-num-seqs 160 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \      # 必须开启
  --block-size 32 \               # 长文本用大块减少管理开销
  --dtype half \
  --swap-space 8                  # 兜底，防止OOM
```

### 模板C：视觉模型（VL）

```bash
vllm serve $MODEL \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization **0.60** \  # 红线！
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --max-num-paddings 4096 \
  --enable-chunked-prefill \
  --block-size 16 \
  --dtype half
```

---

## 第四步：启动验证（3分钟）

### 4.1 启动后必做检查

```bash
# 1. 确认参数生效（防止非法参数静默回退）
curl http://localhost:8000/v1/models | grep -E "max_model_len|max_num_seqs"

# 2. 查看实际启动参数
# 日志中搜索："EngineArgs" 确认无警告

# 3. 显存占用检查
nvidia-smi
# 目标：占用在 (GPU_Memory * gpu_memory_utilization) 的 ±5% 范围内
```

### 4.2 快速压测验证

```bash
# 使用真实业务长度分布进行压测
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  # ... 你的参数

# 压测命令（示例）
python benchmark_serving.py \
  --backend vllm \
  --dataset-name random \
  --model $MODEL \
  --num-prompts 100 \
  --random-range-ratio 0.5 \
  --random-input-len $P80_prompt_len \
  --random-output-len $P80_completion_len
```

### 4.3 关键指标检查

| 指标 | 检查命令/位置 | 通过标准 |
| --- | --- | --- |
| 显存峰值 | `nvidia-smi` 或监控 | < `GPU_Memory * 0.95` |
| KV缓存使用率 | 日志 `vllm_cache_usage` | < 90% |
| 抢占次数 | 日志 `num_preemptions_total` | \= 0（启动初期） |
| P99延迟 | 压测结果 | < 业务SLA |
| 吞吐量 | 压测结果 | \> `Target_QPS` |

---

## 第五步：监控告警配置（部署后）

### 5.1 必设告警项

```yaml
# Prometheus规则示例
- alert: VLLM_Cache_Usage_High
  expr: vllm:gpu_cache_usage_perc > 0.88
  for: 5m
  action: 降低max_num_seqs或增加显存

- alert: VLLM_Preemption_High  
  expr: rate(vllm:num_preemptions_total[5m]) > 0.1
  for: 1m
  action: 立即降低max_num_seqs

- alert: VLLM_GPU_Memory_OOM_Risk
  expr: vllm:gpu_memory_usage > 0.95
  for: 1m
  action: 检查是否有内存泄漏，考虑重启
```

### 5.2 日常巡检

```bash
# 每日检查命令
curl -s http://localhost:8000/metrics | grep -E "cache_usage|preemptions|gpu_memory"
```

---

## 现场问题速查表

| 现象 | 可能原因 | 现场修复 |
| --- | --- | --- |
| 启动即OOM | `max_model_len` 过大或 `gpu_memory_utilization` 过高 | 1\. 降低 `max_model_len` 到8192   2\. VL模型确保≤0.60   3\. 增大 `block-size` 到32 |
| 吞吐量不达标 | `max_num_seqs` 太小或batch效率低 | 1\. 逐步增加 `max_num_seqs` 到192   2\. 增大 `max_num_batched_tokens`   3\. 添加 `--scheduler-delay-factor 0.5` |
| P99延迟极高 | 长请求阻塞或抢占严重 | 1\. 开启 `--enable-chunked-prefill`   2\. 降低 `max_num_seqs` 减少抢占   3\. 检查 `num_preemptions_total` |
| 短请求被饿死 | 连续批处理中长请求垄断 | 1\. 使用 `--preemption-mode swap`   2\. 降低 `max_model_len` 截断长尾   3\. 设置优先级队列 |
| 显存够但报OOM | 视觉模型临时缓冲或碎片 | 1\. 降低 `gpu_memory_utilization` 到0.6   2\. 增大 `block-size` 减少块数   3\. 重启服务清理碎片 |
| 参数改了不生效 | 非法参数静默回退 | 1\. 检查启动日志警告   2\. 用 `/v1/models` 接口验证   3\. 确认参数名拼写正确 |

---

## 极简记忆口诀

```markdown
长度按业务，别追模型极值
并发找拐点，192先试试
视觉卡死线，零点六别越
长文必分块，chunked-prefill
显存留余量，监控保平安
```

---

## 附录：常用模型参考配置

| 模型 | GPU | 推荐配置 |
| --- | --- | --- |
| Qwen3-8B | 4090/24GB | `--max-model-len 8192 --max-num-seqs 192 --gpu-memory-utilization 0.90` |
| Llama3-70B | 2×A100 | `--tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 128` |
| Qwen3-VL-7B | A10/24GB | `--max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.60` |
| DeepSeek-R1-32B | 4×4090 | `--tensor-parallel-size 4 --max-model-len 16384 --enable-chunked-prefill` |

---

**最后提醒** ：这份指南是起点，不是终点。每次业务流量变化后，重新执行 **第一步** 收集数据，持续优化。
