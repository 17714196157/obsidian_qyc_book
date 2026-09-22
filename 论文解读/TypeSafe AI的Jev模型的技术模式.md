项目地址: https://github.com/sugarforever/tryjev


### 🧠 Jev 的核心模式是什么？
Jev 的核心思路是**放弃逐字生成文本，直接输出结构化决策和校准概率**。它的技术模式可概括为以下几点：
*   **并行单次前向计算**：不进行逐 Token 的自回归解码，而是在一次前向传播中并行输出所有答案，这使其延迟极低（70-500ms）。
*   **类型化输出**：只接受预定义的结构化问题（如 `Choice`、`Score`、`Noul`），并从预设选项中选择答案，从机制上避免了生成错误格式或无关文本。
*   **校准概率**：每个答案都附带经过校准的置信度概率，使自动化系统能根据概率阈值进行决策。
*   **专用训练方法（RLCD）**：TypeSafe 使用自研的“面向校准决策的强化学习”（RLCD）来训练模型，优化目标是输出“诚实的概率”，而非人类偏好。

### 🛠️ 如何在 Qwen开源模型上实现？
目前社区的开源实现主要分为三类，你可以根据自己的技术背景和需求选择：

| 路径                | 核心思路                                                                | 代表项目                               | 适合场景                          |
| :---------------- | :------------------------------------------------------------------ | :--------------------------------- | :---------------------------- |
| **最简单：Logits 读取** | 直接利用现成 Qwen 模型，在它准备生成答案时“截住”，读取选项对应 token 的 logits 并转换为概率。**无需训练**。 | `SemIf`、`OpenJev`、`mini-Jev`       | 快速验证想法，或作为轻量级决策层。             |
| **进阶：微调 + 决策头**   | 在 Qwen 基础上，通过 LoRA 微调并添加一个专门的“决策头”，训练模型输出校准后的概率分布。                  | `Kev`、`JevForge`、`reflex`、`litjev` | 追求更好的准确率和校准质量，适合有训练资源的场景。     |
| **深入：完整 RLCD 训练** | 在 Qwen 上复现 Jev 的 RLCD 训练方法，使用强化学习来优化决策和概率校准。                        | `decision-head-rlcd`、`Open-Jev`    | 希望深入研究 Jev 的训练机制，或对校准质量有极致要求。 |

### 💡 关键实现步骤
如果你想自己动手，核心步骤通常包括：
1.  **选择基座模型**：从 Qwen 系列中选择一个合适的模型（如 Qwen3-4B、Qwen3.5-9B 等）。较小的模型（如 0.8B）也可以在消费级 GPU 或 CPU 上运行和微调。
2.  **定义决策头**：在模型输出层后添加一个轻量级的分类头（如几层 GELU），将隐藏状态映射到每个候选选项的得分上。
3.  **准备决策数据**：将你的任务构建成（状态，问题，候选选项）的形式。可以参考 JevForge 的数据集格式。
4.  **训练与校准**：使用交叉熵损失和 **Brier 损失**进行微调，并引入**温度缩放**等方法来校准输出的概率，使其更可靠。
5.  **推理优化**：利用 KV-Cache 和批处理技术，实现一次前向计算为同一状态下的多个问题并行打分，这是保证低延迟的关键。


### 实现代码示例
1. **最简单：Logits 读取**
```python
from openai import OpenAI
import torch
import torch.nn.functional as F
"""
vllm serve /home/qyc/bert/Qwen2-0.5B  --host 0.0.0.0   --port 8000  --dtype half
    --enforce-eager  \
     --max-logprobs 100  \
    --max-num-batched-tokens 8192 \      # 确保足够大的batch
    --max-num-seqs 4 \                   # 明确限制并发
    --enable-chunked-prefill \           # 关键：分块处理长prompt
    --enable-prefix-caching              # 缓存长prompt的KV
"""
# 连接到 vLLM 服务
client = OpenAI(
    base_url="http://192.168.0.181:8000/v1",
    api_key="none",  # vLLM 默认不需要 key，但 openai 库要求
)

state = "Customer writes: my order arrived broken and I need it replaced today."

prompt = f"""You are a decision engine. Read the customer message and choose the most appropriate action.

Customer message: {state}

Options:
A. Refund - Customer wants their money back
B. Replace - Customer wants a replacement item
C. Info - Customer is just asking for information

Answer with a single letter (A, B, or C)."""

# 调用 completions 接口
response = client.completions.create(
    model="/home/qyc/bert/Qwen2-0.5B",
    prompt=prompt,
    max_tokens=1,
    temperature=0.0,
    logprobs=20,  # 请求 top-100 logprobs
)

# 提取 logprobs
logprobs_data = response.choices[0].logprobs
# logprobs_data.top_logprobs 是一个列表，每个元素对应一个生成 token
top_logprobs = logprobs_data.top_logprobs[0]  # dict: {token_str: logprob}

# 选项映射
options = {"A": "refund", "B": "replace", "C": "info"}

# 提取选项概率
option_logprobs = {}
for letter in options.keys():
    # 注意：API 返回的是 token 字符串，需要处理可能的空格
    if letter in top_logprobs:
        option_logprobs[letter] = top_logprobs[letter]
    elif f" {letter}" in top_logprobs:  # 带空格的版本
        option_logprobs[letter] = top_logprobs[f" {letter}"]
    else:
        option_logprobs[letter] = -9999.0
        print(f"警告: 选项 {letter} 不在 top-100 中")

# 转换为张量并 softmax
option_lp_tensor = torch.tensor([option_logprobs[l] for l in options.keys()])
option_probs = F.softmax(option_lp_tensor, dim=-1)

print("决策结果 (OpenAI API 方案):")
for i, (letter, meaning) in enumerate(options.items()):
    print(f"  {letter}. {meaning:8s}  ->  概率: {option_probs[i].item():.4f}")

best_idx = torch.argmax(option_probs).item()
best_letter = list(options.keys())[best_idx]
print(f"\n>>> 最终决策: {options[best_letter]} (选项 {best_letter})")
```