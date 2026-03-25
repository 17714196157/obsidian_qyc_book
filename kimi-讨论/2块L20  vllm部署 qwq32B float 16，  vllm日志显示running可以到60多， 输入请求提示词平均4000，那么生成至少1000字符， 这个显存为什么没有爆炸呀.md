

这是一个关于大模型推理显存占用的技术问题。让我分析一下这个情况：

## 核心原因：KV Cache 动态分配 + 分页管理

vLLM 使用 **PagedAttention** 机制，显存占用不完全取决于 `max_model_len` 或理论计算值，而是**实际运行时的动态分配**。

### 你的配置分析

| 参数 | 数值 |
|------|------|
| 模型 | QwQ-32B |
| 精度 | FP16 |
| GPU | 2× L20 (48GB × 2 = **96GB**) |
| 运行显示 | 60+ requests running |
| 平均输入 | 4000 tokens |
| 生成长度 | 1000+ tokens |

### 为什么没爆显存？

#### 1. **vLLM 的 PagedAttention 机制**
```python
# 不是预分配，而是按需分配
- 传统方式：为每个请求预分配 max_seq_len 的 KV cache
- vLLM方式：像操作系统内存一样分页，用多少分配多少
```

#### 2. **实际计算公式**
```
显存占用 ≈ 模型权重 + KV Cache + Activation

模型权重：32B × 2字节 = 64GB (2卡分摊 = 32GB/卡)
KV Cache：num_layers × num_heads × head_dim × seq_len × batch_size × 2 × 2字节
```

#### 3. **关键：60+ 是并发数，不是 batch size**
- vLLM 使用 **Continuous Batching**（持续批处理）
- 这 60+ 请求**不是同时前向传播**，而是：
  - 有的在处理 prefill（输入阶段）
  - 有的在生成 token（decode 阶段）
  - 有的可能已经生成完毕在等待输出
  - 实际 GPU 上同时计算的 batch 可能只有 **4-16** 个

#### 4. **KV Cache 实际占用估算**
```python
# QwQ-32B 配置 (假设标准 transformer)
layers = 64
heads = 40
head_dim = 128
batch_size = 60  # 假设全部在跑
seq_len = 5000   # 4000输入 + 1000生成

# KV Cache = 2(K+V) × layers × heads × head_dim × seq_len × batch × 2字节
= 2 × 64 × 40 × 128 × 5000 × 60 × 2 / 1e9
≈ 393 GB  # 理论值，但这是连续存储的情况
```

**实际 PagedAttention 优化后：**
- 只有**活跃 token** 才占用显存
- 生成的 token 是**逐 token 分配**
- 不同序列长度**共享/复用**物理块
- 实际可能只有 **20-40GB** KV Cache

#### 5. **L20 的 48GB 显存分布**
```
单卡 L20 (48GB):
├── 模型权重: ~32GB (2卡 TP 后)
├── KV Cache: ~12-15GB (动态)
├── Activation: ~1-2GB
└── 预留/碎片: ~2-4GB
```

### 验证命令

你可以运行以下命令查看实际显存分布：

```bash
# 查看 vLLM 内存分析
curl http://localhost:8000/metrics | grep vllm

# 或启动时开启详细日志
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/QwQ-32B \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --enable-chunked-prefill \  # 关键：分块预填充
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95
```

### 为什么能跑 60+ 的关键配置

| vLLM 参数 | 作用 |
|-----------|------|
| `--enable-chunked-prefill` | 长输入分块处理，降低峰值显存 |
| `--max-num-batched-tokens` | 限制每批处理的 token 数 |
| `--max-num-seqs` | 最大序列数，但受显存动态限制 |
| `--gpu-memory-utilization` | 通常 0.9，预留 10% 防止 OOM |
| `--swap-space` | 可把部分 KV cache 换到 CPU |

### 风险提示 ⚠️

虽然现在没爆，但存在隐患：

1. **显存碎片**：长期运行后可能出现 OOM
2. **生成长度不均**：如果某些请求生成远超 1000 token，可能突然爆显存
3. **并发突增**：如果 60+ 同时进入 prefill 阶段，可能峰值不足

建议监控：
```bash
watch -n 1 nvidia-smi
# 关注显存波动，特别是已分配 vs 保留的区别
```

### 总结

> **60+ running 是调度层面的并发，不是计算层面的 batch size。vLLM 的 PagedAttention + Continuous Batching 让这 60 个请求像"流水线"一样共享 GPU，实际同时计算的只有几到十几个，所以 96GB 显存够跑 QwQ-32B。**

如果这 60 个请求真的**同时做 prefill**（比如同时收到 60 个 4000 token 的长文本），那大概率会 OOM。但实际场景是交错到达的，vLLM 的调度器会智能控制。