## 📋 问题总结

这是一个 **vLLM 0.4.3** 版本的 Bug：**并发请求时，greedy 解码（temperature=0）的结果会受到其他采样请求的影响**，导致结果不一致。
### 核心问题
- 当 `temperature=0`（贪婪解码）时，结果应该是**完全确定性的**
- 但并发执行时，greedy 请求的结果会被其他带有采样参数（如 `seed`、`top_k`）的请求"污染"
- 表现为同样的 greedy 请求返回了不同的结果（如 `old man` vs `young woman`）
## 📊 检测方法

| 现象                        | 结论         |
| ------------------------- | ---------- |
| 所有 greedy 请求返回**完全相同**的结果 | ✅ 无此 Bug   |
| greedy 请求返回**2种或以上**不同结果  | ❌ 存在此 Bug  |
| greedy 结果与单请求基准**不一致**    | ❌ 存在并发污染问题 |

---

## 🔍 如何检测你的 vLLM 是否有同样问题
对比测试: 单请求 greedy 和并发 greedy 的结果
广福环境测试结果 ： 
![[工作交付文档/assets/vllm-issues5607-单请求vs并行请求一致性核对/dc6111818b2c7f88c1d715f90313574f_MD5.png]]

![[工作交付文档/assets/vllm-issues5607-单请求vs并行请求一致性核对/81178cb1439d5ed003b09a902898ed9d_MD5.png]]

对比测试:  检测并发请求时 greedy 解码是否一致  生成100个字符
10个采样解码请求和10个贪心解码请求同时并发，查看是否存在干扰。
广福环境测试结果 ： 
![[工作交付文档/assets/vllm-issues5607-单请求vs并行请求一致性核对/a085b86f78f74daf2650d37351e8d67f_MD5.png]]

#### 固定seed=42，期望并行请求结果一致

![[工作交付文档/assets/vllm-issues5607-单请求vs并行请求一致性核对/fb6a3ae287099f0c7311fcc6f47be817_MD5.png]]


 **根本原因**
这是 **vLLM 的 CUDA kernel 非确定性（nondeterminism）** 导致的，即使设置了 `temperature=0` 和 `seed`：
1. **浮点运算的非结合性（核心原因）**
GPU 并行计算时，浮点数加法的顺序会影响最终结果（`(a+b)+c ≠ a+(b+c)`）。当 batch size 或请求组合方式变化时，kernel 的 reduction 顺序不同，导致 logits 出现微小差异[](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)。这些差异会累积，最终导致生成 token 不同，生成长度也可能不同。
2. **Batching 动态调度**
vLLM 的 continuous batching 会根据并发请求动态组合 batch。并行请求时，请求可能被分到不同 batch 或不同顺序处理，触发不同的 kernel 执行路径
3. **Kernel 自动调优（Triton autotuner）**
某些 kernel（如 `tl.cumsum`）的 Triton autotuner 会根据运行时选择不同配置（如 `BLOCK_SIZE_H=1` vs `2`），这些配置会产生数值差异
4. **精度问题**
使用 `bfloat16` 时数值稳定性更差，更容易出现不一致

### 我观察到都是只有第一个和后续请求的不一样
具体表现
- **第一个请求**：走 eager 模式或部分捕获路径，结果可能不正确
- **第二个及以后**：走完整的 CUDA Graph 路径，结果一致且正确

| 场景           | 原因                                       |
| ------------ | ---------------------------------------- |
| **严格串行**     | 第 1 个请求捕获 CUDA Graph，第 2+ 个走 Graph 路径    |
| **并行（同时发送）** | 第一个被调度的请求触发捕获，其他请求等待或部分参与，导致状态混乱         |
| **连续批量**     | 动态 batching 使得每个 batch 的第一个请求都可能触发新的形状捕获 |

##### 实际解决方案
1. **使用模型自身的思考长度控制（推荐）**
在请求时通过系统提示限制
```python
system_prompt = """你是一个高效的助手。请遵循以下规则：
1. 思考过程要简洁明了，不要过度展开
2. 直接抓住问题核心，避免冗余推理
3. 总输出控制在500字以内"""
```
  
2. max_tokens 参数设置注意点
```bash
vllm serve QwQ-32B --max_model_len 20000
```
`max_model_len` 是上下文总长度限制（输入+输出），会覆盖模型 config 中的设置

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/QwQ-32B")
sampling_params = SamplingParams(
    max_tokens=1024,  # 在这里设置！
    temperature=0.6
)
outputs = llm.generate(prompts, sampling_params)

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[{"role": "user", "content": "你的问题"}],
    max_tokens=1024,  # ← 在这里控制生成长度！
    temperature=0.6
)
```
推理模型可能在 `<think>...` 标签内的内容**不计入** `max_tokens` 限制（取决于具体实现），只有 `</thinking>` 后的正式答案受限制

3. 固定seed=42， temperature=0

4. max-seq-len-to-capture 参数优化
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

   5.enable-chunked-prefill  # 关键：长输入分块处理，长输入不阻塞
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

6. 自投机
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
```
7. 使用 `--enforce-eager` 模式（可能有效）
```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --enforce-eager  # 禁用 CUDA graph，可能避免状态污染
```

---

### 关于参数对模型输出一致性的影响

| 公共信息：模型FP8<br>数据集97份病案中进入诊断细化的54份病案              | 2次编码差异的数量<br>（55% 30/54 是8次测试都完全一致的） | 耗时   |
| ------------------------------------------------ | ------------------------------------ | ---- |
| 温度0.9,top_p 0.95                                 | 25.9% (14/54)                        | 271s |
| 温度0.0,top_p 1, seed=42                           | 18% （10/54）                          | 280s |
| 温度0.0 --no-enable-prefix-caching                 | 11%（6/54）                            | 310s |
| 温度0.0 --no-enable-prefix-caching --enforce-eager | 12.9%（7/54）                          | 270s |
结论： **调整参数可以降低一半的不一致性，有影响但并非决定性的**，
可能原因：
- 0.8.4 版本已知问题 **CUDA Graph 是最大不确定性来源**
即使设置 `seed=42`，CUDA Graph 的 capture/replay 机制会引入非确定性。--enforce-eager是必须的。
- Prefix Caching 的数值误差
0.8.x 版本的自动前缀缓存（默认开启）在计算 KV Cache 时可能引入微小差异，建议关闭：--no-enable-prefix-caching


**下一步计划： 0.18.0 vllm库支持更多配置，强制贪心（确定性最高）
脚本检查是否结果稳定一致
```python
import openai
import hashlib

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def test_determinism(prompt, rounds=5):
    results = []
    for i in range(rounds):
        resp = client.chat.completions.create(
            model="your_model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1.0,
            top_k=1,
            seed=42,
            max_tokens=100,
        )
        text = resp.choices[0].message.content
        results.append(hashlib.md5(text.encode()).hexdigest()[:8])
        print(f"Round {i+1}: {text[:50]}... (hash: {results[-1]})")
    
    print(f"\nAll same: {len(set(results)) == 1}")

test_determinism("请用一句话解释什么是深度学习", rounds=3)

Token: Once
Logprob: -0.8439942598342896
Top logprobs: []
```
**贪婪解码的特征**：
- `top_logprobs` 列表长度为 1（只有选中的那个）
- 或者 logprob 接近 0（概率接近 1.0）