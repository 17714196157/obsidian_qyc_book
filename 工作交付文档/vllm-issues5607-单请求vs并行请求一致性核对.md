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
对比测试: 单请求 greedy 和并发 greedy 的结果  生成2000个字符
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


官方确认的 Issue
这是一个已被官方确认的 Bug：
> **Issue #19403** [Bug]: Issue of Unstable Output for Identical Queries
> - 串行请求时，第一个请求始终返回错误输出
> - 并发请求时，错误随机出现
> - **设置 `--enforce-eager=True` 后恢复正常**
>     
> **Issue #17832** [Bug]: First API call differs from subsequent identical calls
> - 使用 `temperature=0` 时，第一个请求与后续请求输出不同
> - 从第二个请求开始保持一致



### 方法二：检查 vLLM 版本

```bash
# 查看当前 vLLM 版本
python -c "import vllm; print(vllm.__version__)"

# 或使用 pip
pip show vllm
```

**受影响版本**：根据 Issue，问题出现在 **vLLM 0.4.3** 及相近版本。

### 方法三：查看 Issue 修复状态

检查该 Issue 是否已关闭以及修复版本：

```bash
# 查看 vLLM 最新版本
pip index versions vllm
```

---

## 🛠️ 解决方案

如果检测发现你有同样的问题，可以尝试：

### 1. **升级 vLLM**
```bash
pip install --upgrade vllm
```

### 2. **临时规避方案**
- **避免混用**：不要将 greedy 请求（`temperature=0`）和采样请求（`temperature>0`）并发发送到同一实例
- **分离部署**：为 greedy 任务和采样任务分别部署独立的 vLLM 实例
- **串行处理**：对需要确定性的 greedy 请求进行串行处理

### 3. **使用 `--enforce-eager` 模式（可能有效）**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --enforce-eager  # 禁用 CUDA graph，可能避免状态污染
```

---
