## 一）enable-chunked-prefill  
enable-chunked-prefill  # 关键：长输入分块处理，长输入不阻塞
	传统 Prefill 的瓶颈
```
Batch 1: [短请求A: 100 tokens]  [短请求B: 100 tokens]  → 一起prefill，很快
Batch 2: [长请求C: 4000 tokens]                        → 阻塞整个队列！
         ↑ 这4000 tokens的prefill计算期间，GPU无法处理其他请求
```
**关键问题**：长输入的 Prefill 是**原子操作**，不可分割，导致：
- 短请求被长请求阻塞（Head-of-Line Blocking）
- GPU 在 prefill 期间无法穿插做 decode
- 整体吞吐下降
Chunked-Prefill 的核心思想**把长输入切成小块，穿插执行**
```plain
传统方式:
[CCCCCCCCCCCCCCCC]  长请求C的4000 token prefill（阻塞16ms）
[DDDD]               短请求D的decode
[DDDD]               
[DDDD]               

Chunked-Prefill:
[CCCC]  chunk1 (1000 tokens)  ← 4ms
[DDDD]  decode短请求A
[CCCC]  chunk2 (1000 tokens)
[DDDD]  decode短请求B  
[CCCC]  chunk3
[DDDD]  decode短请求C
[CCCC]  chunk4
[DDDD]  decode...
```
**效果**：长请求的 prefill 不再阻塞队列，decode 可以穿插执行，提升 GPU 利用率。


## 二）max-seq-len-to-capture 参数优化
   **CUDA Graph** 是 NVIDIA GPU 的一项优化技术：
- 正常情况下，GPU 运算需要 CPU 一步步下发指令（kernel launch），CPU 和 GPU 频繁通信有开销
- CUDA Graph 把一系列 GPU 操作**提前录制**下来，打包成一个"执行图"
- 运行时**一次性提交**整个图，减少 CPU 开销，大幅提升推理速度
```
传统模式：CPU → 下发kernel → GPU执行 → CPU等待 → 下发下一个kernel → ...
CUDA Graph：CPU → 提交整个图 → GPU连续执行所有操作 → 完成
```

| 设置                              | 含义                            |
| ------------------------------- | ----------------------------- |
| `--max-seq-len-to-capture 1024` | 只有长度 ≤1024 的序列走 CUDA Graph 优化 |
| `--max-seq-len-to-capture 8192` | 长度 ≤8192 的序列都走优化              |
| **你的请求情况**                      | **效果**                        |
| 输入+输出 ≤ 8192 tokens             | 走 CUDA Graph，**速度最快**         |
| 输入+输出 > 8192 tokens             | 回退到普通模式，**速度变慢**              |
### 三）**Continuous Batching**
vLLM 没有固定的 `batch_size`，而是 **Continuous Batching**，每轮迭代都动态重组 batch。
**核心思想**：当一个序列生成完一个 token 后，如果还有其他 token 要生成，继续留在批次中；同时，新的请求可以**立即加入**正在运行的批次，无需等待当前批次完全结束。

| 概念                       | 含义               | 影响                   |
| ------------------------ | ---------------- | -------------------- |
| `max_num_batched_tokens` | 每批次最大token数      | 提高prefill效率，但增加单次计算量 |
| `max_model_len`          | 单请求最大token数      | 支持更长上下文，但KV缓存块数线性增加  |
| `max_num_seqs`           | **调度层面**的最大并发序列数 | 提高，但会增加调度开销          |

```bash
vllm serve /home/qyc/bert/Qwen2-0.5B  --host 0.0.0.0   --port 8000  --dtype half  --max-num-seqs 3  --uvicorn-log-level debug  > vllm.log 2>&1
启动日志
    Avg prompt throughput: 1254.3 tokens/s
    Avg generation throughput: 892.1 tokens/s
    Running: 12 reqs,    # ← 当前正在生成的请求数
    Swapped: 0 reqs,     # ← 被交换到CPU的请求（内存不足时）
    Pending: 3 reqs,     # ← 等待加入批次的请求
    
vLLM 的 Scheduler 决策日志
export VLLM_LOGGING_LEVEL=DEBUG



```
假设场景：
- 请求 A：prompt 100 tokens，需生成 50 tokens
- 请求 B：在 A 生成到第 10 个 token 时到达
- 请求 C：在 A 生成到第 25 个 token 时到达
```
Step 0:  [A_prefill(100)]           # A 的 prefill 阶段
Step 1:  [A_gen(1)]                 # A 生成第1个token
...
Step 10: [A_gen(10), B_prefill(80)] # B加入！A继续生成，B做prefill
Step 11: [A_gen(11), B_gen(1)]      # 一起生成
...
Step 25: [A_gen(25), B_gen(15), C_prefill(120)] # C加入！
```

## 四） speculative 自投采样
```
# 方案A：独立 draft 模型（推荐）
vllm serve Qwen/QwQ-32B \
    --speculative-model Qwen/Qwen2.5-0.5B-Instruct \  # 小模型做draft
    --num-speculative-tokens 5 \
    --max-seq-len-to-capture 12000

# 方案B：自投机（无额外模型，用自身做draft）
vllm serve Qwen/QwQ-32B \
    --speculative-max-model-len 16384 \
    --num-speculative-tokens 3 \
    --max-seq-len-to-capture 12000
