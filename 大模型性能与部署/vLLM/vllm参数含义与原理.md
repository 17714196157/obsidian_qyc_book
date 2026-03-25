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

## 三） speculative 自投采样
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
