项目地址: https://github.com/sugarforever/tryjev
TypeSafe AI 的官网： https://console.typesafe.ai/
**awesome-jev**（GitHub: `yibie/awesome-jev`）是维护最全面的 Jev 生态列表


![[file-20260923162227210.png]]
### 🧠 Jev 的核心模式是什么？
Jev 的核心思路是**放弃逐字生成文本，直接输出结构化决策和校准概率**。它的技术模式可概括为以下几点：
*   **并行单次前向计算**：不进行逐 Token 的自回归解码，而是在一次前向传播中并行输出所有答案，这使其延迟极低（70-500ms）。
*   **类型化输出**：只接受预定义的结构化问题（如 `Choice`、`Score`、`Noul`），并从预设选项中选择答案，从机制上避免了生成错误格式或无关文本。只在候选选项的 token 集合上做 Softmax，而不是在整个词表上。这保证了输出必然在预设的选项内，从数学上杜绝了格式错误或“幻觉”。
*  **KV-Cache 复用**：对于共享同一个 `state`（上下文）的多个独立问题，可以只计算一次上下文的 KV-Cache，然后并行地对每个问题进行评分，极大提升效率。
*   **校准概率**：每个答案都附带经过校准的置信度概率，使自动化系统能根据概率阈值进行决策。
*   **专用训练方法（RLCD）**：TypeSafe 使用自研的“面向校准决策的强化学习”（RLCD）来训练模型，优化目标是输出“诚实的概率”，而非人类偏好。
==Jev 的速度来自并行约束解码（parallel constrained decoding），而非特定的模型训练。理论上，任何推理引擎都可以通过约束解码暴露 Jev 风格的 API。==
传统 LLM 通过自回归方式逐 token 生成文本，而 Jev 式的“System One”推理则完全不同：
- **传统方式**：`[Prompt] -> [自回归生成] -> [输出文本 "The answer is 90"] -> [解析文本]`
- **Jev 式**：`[Prompt + 选项A/B/C] -> [单次前向传播] -> [读取选项A/B/C的logits] -> [Softmax归一化] -> [直接得到概率分布]`

### 🛠️ 如何在 Qwen开源模型上实现？
目前社区的开源实现主要分为三类，你可以根据自己的技术背景和需求选择：

| 路径                | 核心思路                                                                | 代表项目                               | 适合场景                          |
| :---------------- | :------------------------------------------------------------------ | :--------------------------------- | :---------------------------- |
| **最简单：Logits 读取** | 直接利用现成 Qwen 模型，在它准备生成答案时“截住”，读取选项对应 token 的 logits 并转换为概率。**无需训练**。 | `SemIf`、`OpenJev`、`mini-Jev`       | 快速验证想法，或作为轻量级决策层。             |
| **进阶：微调 + 决策头**   | 在 Qwen 基础上，通过 LoRA 微调并添加一个专门的“决策头”，训练模型输出校准后的概率分布。                  | `Kev`、`JevForge`、`reflex`、`litjev` | 追求更好的准确率和校准质量，适合有训练资源的场景。     |
| **深入：完整 RLCD 训练** | 在 Qwen 上复现 Jev 的 RLCD 训练方法，使用强化学习来优化决策和概率校准。                        | `decision-head-rlcd`、`Open-Jev`    | 希望深入研究 Jev 的训练机制，或对校准质量有极致要求。 |

### 关键实现步骤
如果你想自己动手，核心步骤通常包括：
1.  **选择基座模型**：从 Qwen 系列中选择一个合适的模型（如 Qwen3-4B、Qwen3.5-9B 等）。较小的模型（如 0.8B）也可以在消费级 GPU 或 CPU 上运行和微调。
2.  **定义决策头**：在模型输出层后添加一个轻量级的分类头（如几层 GELU），将隐藏状态映射到每个候选选项的得分上。
3.  **准备决策数据**：将你的任务构建成（状态，问题，候选选项）的形式。可以参考 JevForge 的数据集格式。
4.  **训练与校准**：使用交叉熵损失和 **Brier 损失**进行微调，并引入**温度缩放**等方法来校准输出的概率，使其更可靠。
5.  **推理优化**：利用 KV-Cache 和批处理技术，实现一次前向计算为同一状态下的多个问题并行打分，这是保证低延迟的关键。

#### tips：
- 1. ✅ `system_one_adapter` 是官方 SDK，但定位是“对照组”
`system_one_adapter` 确实是 TypeSafe AI 官方发布的 Python 包，在官方 `awesome-jev` 列表中明确标注为 **“Official drop-in `TypeSafeClient` replacement backed by LLM APIs”**。它的用途是：**用普通 LLM（通过 OpenAI/Anthropic API）来模拟 Jev 的输入输出接口，从而让你能在同一套问题上对比“Jev vs 普通 LLM”的效果、成本和速度。** 它**故意不**使用 Jev 的 logits 读取机制，因为它要模拟的是“用传统 LLM 做同样的事”会怎样，而不是复现 Jev 本身。

- 2. typesafe_sdk是云服务api配套的SDK，请求的接口是定制的格式，vllm占时不支持。


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

2. 官方sdk 
docker pull docker.1ms.run/razorback16/openjev
docker pull docker.1panel.live/razorback16/openjev
docker pull docker.m.daocloud.io/razorback16/openjev
```python
import os
from typesafe_sdk import Choice, TypeSafeClient
# 部署方式：官方提供 Docker 镜像（razorback16/openjev），把 vLLM 和 API server 打包在同一个容器里。

# 必须在 import typesafe_sdk 之前设置
os.environ["TYPESAFE_BASE_URL"] = "http://192.168.0.181:8000"
os.environ["TYPESAFE_API_KEY"] = "none"

# 1. 设置你的 API Key（如果你连接的是官方服务或兼容服务）
# 也可以直接通过参数传入：TypeSafeClient(api_key="your-key")
# os.environ["TYPESAFE_API_KEY"] = "your-key-here"

state = "Customer writes: my order arrived broken and I need it replaced today."
# 连接到 vLLM 服务
client = TypeSafeClient(
     base_url="http://192.168.0.181:8000/v1",
     api_key="none",  # vLLM 默认不需要 key，但 openai 库要求
)


# 2. 使用上下文管理器创建客户端，确保连接正确关闭
with TypeSafeClient() as client:
    # 3. 调用 system_one，传入 state 和结构化的问题
    response = client.system_one(
        state=state,
        questions={
            "route": Choice(
                instructions="Which team handles this?",
                criteria={
                    "refund": "Customer wants their money back",
                    "replace": "Customer wants a replacement item",
                    "info": "Customer is just asking for information",
                },
            ),
        },
    )

# 4. 直接从 response 中读取结果
route_answer = response.choices["route"]
print("决策结果 (TypeSafe SDK 方案):")
print("=" * 60)
print(f"\n  选择: {route_answer.choice}")
print(f"  概率分布: {route_answer.probabilities}")

# 概率已经是归一化后的结果，直接取最大值即可
best_choice = max(route_answer.probabilities, key=route_answer.probabilities.get)
best_prob = route_answer.probabilities[best_choice]
print(f"\n>>> 最终决策: {best_choice}  (置信度: {best_prob:.4f})")

# 显示用量信息
print(f"\n--- 调试信息 ---")
print(f"  输入 Token: {response.usage.input_tokens}")
print(f"  输出 Token: {response.usage.output_tokens}")
```



### 开源社区情况
#### laya
项目地址： [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
Laya（Convai Innovations，Apache 2.0，非自回归 ModernBERT/mmBERT 决策引擎，choice/score/noul 三原语与 Jev 接口一致）是国内讨论和实测最多的复现对象。本质是"待微调的快速底座"而非开箱即用引擎；512/1024 token 的上下文远小于 Jev 的 64K；

代码里 `Router(preload=True)` 内部会加载 **Laya 的三个 checkpoint**，它们都在 HuggingFace 上，repo 是 [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)：

| checkpoint             | 底层编码器            | 参数量  | context | 用途                  | Router 里的名字         |
| ---------------------- | ---------------- | ---- | ------- | ------------------- | ------------------- |
| `laya`                 | ModernBERT-large | 421M | 512     | 英文                  | `"english"`         |
| `laya-multilingual`    | mmBERT-base      | 322M | 1024    | 100+ 语言             | `"multilingual"`    |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024    | typed-decisions 工作流 | `"typed-decisions"` |

##### 代码示例
- **1)召回重排序rank**
> [!note]- 📄 rerank_batch.py — Laya 批量重排器（点击展开 / 收起）
> ```python
> # rerank_batch.py
> # -*- coding: utf-8 -*-
> """
> Laya 批量重排器
> - 一次前向评估多个文档（把 N 个文档放进同一个 state，对每个文档问一个 relevance 问题）
> - 指定本地下载好的 laya-multilingual 模型
> """
>
> from typing import List, Dict
> import math
> import laya
>
> # ============================================================
> # 1. 配置区：改这里
> # ============================================================
>
> # 本地模型路径。按你的实际目录结构二选一：
> #   A) 整仓下载（仓根下有 multilingual/ 子目录）：
> LOCAL_MODEL_PATH = "./laya_model"
> SUBFOLDER = "multilingual"
> #
> #   B) 只下载了 multilingual 子目录（路径那层直接有 config.json）：
> # LOCAL_MODEL_PATH = "./laya_model/multilingual"
> # SUBFOLDER = None
>
> DEVICE = "cuda"          # 无 GPU 改成 "cpu"
>
> # 每次前向最多评估多少个文档。
> # README 提示：选项数太多会挤占 head 预算导致精度下降，建议 8~10。
> BATCH_SIZE = 8
>
> # 单篇文档最大字符数（粗截断，避免挤爆 state 预算）。
> MAX_DOC_CHARS = 800
>
> # head 预算：query + 选项提示词 + 相关性等级描述都算在这里。
> # 文档多或 query 长时适当调大。
> HEAD_MAX_LEN = 384
>
> # 总序列长度。多语言 mmBERT 默认 1024，encoder 上限 8192。
> MAX_LEN = 1024
>
> # 相关性等级描述（score 类型会在这个有序刻度上输出期望值 0.0 ~ 3.0）
> RELEVANCE_CRITERIA = [
>     "完全不相关",
>     "弱相关，只有边缘信息",
>     "中等相关，包含部分有用信息",
>     "高度相关，直接回答查询",
> ]
>
> # ============================================================
> # 2. 加载本地模型
> # ============================================================
> def load_agent():
>     kwargs = {"device": DEVICE}
>     if SUBFOLDER:
>         agent = laya.load(LOCAL_MODEL_PATH, subfolder=SUBFOLDER, **kwargs)
>     else:
>         agent = laya.load(LOCAL_MODEL_PATH, **kwargs)
>
>     # 调整预算（在文档多 / query 长时很重要）
>     agent.cfg["head_max_len"] = HEAD_MAX_LEN
>     agent.cfg["max_len"] = MAX_LEN
>     return agent
>
>
> # ============================================================
> # 3. 批量重排核心
> # ============================================================
> def _truncate(text: str, max_chars: int) -> str:
>     if len(text) <= max_chars:
>         return text
>     # 简单前截断。中文场景如需更精细，可换成按 tokenizer 截断。
>     return text[:max_chars] + "…"
>
> def _build_questions(query: str, n: int) -> Dict:
>     """为 n 个文档各构造一个 relevance 问题。"""
>     return {
>         f"rel_{i}": {
>             "type": "score",
>             "instructions": (
>                 f"给定查询「{query}」，"
>                 f"文档 doc_{i} 与查询的相关程度如何？"
>             ),
>             "criteria": RELEVANCE_CRITERIA,
>         }
>         for i in range(n)
>     }
>
>
> def _predict_batch(agent, query: str, batch: List[Dict]) -> List[float]:
>     """对一个 batch（<= BATCH_SIZE）的文档做一次前向，返回各文档的 relevance 分数。"""
>     n = len(batch)
>     # 把 n 个文档放进同一个 state
>     state = {f"doc_{i}": _truncate(d["text"], MAX_DOC_CHARS)
>              for i, d in enumerate(batch)}
>     questions = _build_questions(query, n)
>
>     res = agent.predict(state, questions)
>     scores = []
>     for i in range(n):
>         ans = res["answers"][f"rel_{i}"]
>         scores.append(float(ans["score"]))
>     return scores
>
>
> def rerank_batch(
>     agent,
>     query: str,
>     docs: List[Dict],
>     top_k: int = 5,
>     batch_size: int = BATCH_SIZE,
> ) -> List[Dict]:
>     """
>     批量重排主函数。
>
>     docs: [{"id": ..., "text": ...}, ...]
>     返回: 按 Laya relevance 分数降序的 top_k，每项带上 laya_score。
>     """
>     if not docs:
>         return []
>
>     # 先记录原始顺序，方便把分数写回
>     indexed = list(enumerate(docs))
>     scored: List[Dict] = []
>
>     # 分批，一次前向处理一批
>     for start in range(0, len(indexed), batch_size):
>         batch_pairs = indexed[start:start + batch_size]
>         batch_docs = [d for _, d in batch_pairs]
>         scores = _predict_batch(agent, query, batch_docs)
>
>         for (orig_idx, doc), s in zip(batch_pairs, scores):
>             scored.append({
>                 **doc,
>                 "orig_index": orig_idx,
>                 "laya_score": s,
>             })
>
>     # 排序：分数降序；分数相同按原顺序稳定
>     scored.sort(key=lambda x: (-x["laya_score"], x["orig_index"]))
>     return scored[:top_k]
>
>
> # ============================================================
> # 4. 测试
> # ============================================================
>
> def main():
>     print("加载本地模型中 …")
>     agent = load_agent()
>     print(f"模型已加载: {LOCAL_MODEL_PATH} / {SUBFOLDER}")
>     print(f"head_max_len={HEAD_MAX_LEN}, max_len={MAX_LEN}, "
>           f"batch_size={BATCH_SIZE}\n")
>
>     query = "Laya 的中文支持怎么样？"
>
>     corpus = [
>         {"id": 1, "text": "Laya-multilingual 支持 100+ 语言，中文属于 mmBERT 覆盖范围。"},
>         {"id": 2, "text": "Laya 英文 checkpoint 在非拉丁文字上会崩，高棉语 0 准确率却 95% 置信度。"},
>         {"id": 3, "text": "vLLM 的 PagedAttention 用于优化 KV cache，提升生成吞吐。"},
>         {"id": 4, "text": "中文场景建议使用 multilingual checkpoint，并先做温度校准。"},
>         {"id": 5, "text": "今天股市大涨，上证指数收涨 2%。"},
>         {"id": 6, "text": "Laya 是 encoder-only 模型，单次前向输出 typed decisions。"},
>         {"id": 7, "text": "Python 3.10 是 Laya 的最低版本要求。"},
>         {"id": 8, "text": "Router 会根据脚本和语言自动在三个 checkpoint 之间路由。"},
>         {"id": 9, "text": "红烧肉的做法：五花肉切块，冷水下锅焯水……"},
>         {"id": 10, "text": "中文分词工具对比：jieba、HanLP、LAC。"},
>     ]
>
>     results = rerank_batch(agent, query, corpus, top_k=5)
>
>     print(f"查询: {query}\n")
>     print("重排结果（top 5）:")
>     print("-" * 70)
>     for r in results:
>         print(f"[{r['id']:>2}]  score={r['laya_score']:.3f}  {r['text']}")
>     print("-" * 70)
>
>
> if __name__ == "__main__":
>     main()
> """
> 模型已加载: ./laya_model / multilingual
> head_max_len=384, max_len=1024, batch_size=8
> 查询: Laya 的中文支持怎么样？
> 重排结果（top 5）:
> ---------------------------
> [ 4]  score=1.643  中文场景建议使用 multilingual checkpoint，并先做温度校准。
> [ 7]  score=1.626  Python 3.10 是 Laya 的最低版本要求。
> [ 1]  score=1.613  Laya-multilingual 支持 100+ 语言，中文属于 mmBERT 覆盖范围。
> [ 8]  score=1.588  Router 会根据脚本和语言自动在三个 checkpoint 之间路由。
> [ 2]  score=1.561  Laya 英文 checkpoint 在非拉丁文字上会崩，高棉语 0 准确率却 95% 置信度。
> """
> ```
> 

- **2) docker部署 laya模型服务server**
docker-compose 构建文件见附件： [docker-compose构建文件](docker-compose构建文件.zip)

> [!note]- 📄 server.py — Laya HTTP 服务（点击展开 / 收起）
> ```python
> # server.py
> # -*- coding: utf-8 -*-
> """
> Laya HTTP 服务
> - 启动时 preload 本地模型，进程常驻
> - 提供 /rerank 批量重排、/classify 分类、/health 健康检查
> - 批处理聚合：把并发请求攒成一批，一次前向处理，提升吞吐
> """
>
> import asyncio
> import time
> from typing import List, Dict, Optional
> from contextlib import asynccontextmanager
>
> from fastapi import FastAPI, HTTPException
> from pydantic import BaseModel, Field
> import laya
>
>
> # ============================================================
> # 1. 配置
> # ============================================================
>
> LOCAL_MODEL_PATH = "/models/laya_model"      # 改成你的本地路径
> SUBFOLDER = "multilingual"             # 中文用 multilingual；单下子目录则设为 None
> DEVICE = "cuda"                        # 无 GPU 改 "cpu"
>
> HEAD_MAX_LEN = 384
> MAX_LEN = 2048
> MAX_DOC_CHARS = 800
>
> # 批处理聚合参数
> MAX_BATCH_SIZE = 8         # 一次前向最多几个文档
> MAX_WAIT_MS = 20           # 攒批最多等这么久
> RERANK_TIMEOUT = 30.0      # 单请求超时（秒）
>
>
> # ============================================================
> # 2. 全局模型（lifespan 里加载，进程常驻）
> # ============================================================
>
> class ModelHolder:
>     agent = None
>     lock = asyncio.Lock()
>
> holder = ModelHolder()
>
>
> @asynccontextmanager
> async def lifespan(app: FastAPI):
>     # ---- 启动 ----
>     print("加载 Laya 模型中 …")
>     t0 = time.time()
>     kwargs = {"device": DEVICE}
>     if SUBFOLDER:
>         agent = laya.load(LOCAL_MODEL_PATH, subfolder=SUBFOLDER, **kwargs)
>     else:
>         agent = laya.load(LOCAL_MODEL_PATH, **kwargs)
>
>     agent.cfg["head_max_len"] = HEAD_MAX_LEN
>     agent.cfg["max_len"] = MAX_LEN
>     holder.agent = agent
>     print(f"模型加载完成，耗时 {time.time() - t0:.2f}s")
>
>     yield
>
>     # ---- 关闭 ----
>     print("释放模型 …")
>     holder.agent = None
>
>
> app = FastAPI(title="Laya Rerank Service", lifespan=lifespan)
>
>
> # ============================================================
> # 3. 请求/响应模型
> # ============================================================
>
> class Doc(BaseModel):
>     id: Optional[str] = None
>     text: str
>
>
> class RerankRequest(BaseModel):
>     query: str
>     documents: List[Doc]
>     top_k: int = Field(default=5, ge=1)
>     batch_size: int = Field(default=MAX_BATCH_SIZE, ge=1, le=32)
>
>
> class RerankItem(BaseModel):
>     id: Optional[str]
>     text: str
>     orig_index: int
>     laya_score: float
>
>
> class RerankResponse(BaseModel):
>     results: List[RerankItem]
>     latency_ms: float
>
>
> class ClassifyRequest(BaseModel):
>     state: Dict[str, str]           # 任意字段，如 {"body": "..."}
>     questions: Dict[str, dict]      # Laya questions schema
>     model: Optional[str] = None     # 预留，多 checkpoint 时用
>
>
> class ClassifyResponse(BaseModel):
>     answers: Dict
>     latency_ms: float
>
>
> # ============================================================
> # 4. 内部：批量重排（线程池执行，避免阻塞 event loop）
> # ============================================================
>
> def _truncate(text: str, max_chars: int) -> str:
>     return text if len(text) <= max_chars else text[:max_chars] + "…"
>
>
> def _relevance_questions(query: str, n: int) -> Dict:
>     return {
>         f"rel_{i}": {
>             "type": "score",
>             "instructions": f"给定查询「{query}」，文档 doc_{i} 与查询的相关程度如何？",
>             "criteria": [
>                 "完全不相关",
>                 "弱相关，只有边缘信息",
>                 "中等相关，包含部分有用信息",
>                 "高度相关，直接回答查询",
>             ],
>         }
>         for i in range(n)
>     }
>
>
> def _predict_batch_sync(agent, query: str, batch: List[Dict]) -> List[float]:
>     n = len(batch)
>     state = {f"doc_{i}": _truncate(d["text"], MAX_DOC_CHARS)
>              for i, d in enumerate(batch)}
>     questions = _relevance_questions(query, n)
>     res = agent.predict(state, questions)
>     return [float(res["answers"][f"rel_{i}"]["score"]) for i in range(n)]
>
>
> def _rerank_sync(agent, query: str, docs: List[Dict],
>                  top_k: int, batch_size: int) -> List[Dict]:
>     indexed = list(enumerate(docs))
>     scored = []
>     for start in range(0, len(indexed), batch_size):
>         batch_pairs = indexed[start:start + batch_size]
>         batch_docs = [d for _, d in batch_pairs]
>         scores = _predict_batch_sync(agent, query, batch_docs)
>         for (orig_idx, doc), s in zip(batch_pairs, scores):
>             scored.append({**doc, "orig_index": orig_idx, "laya_score": s})
>     scored.sort(key=lambda x: (-x["laya_score"], x["orig_index"]))
>     return scored[:top_k]
>
>
> # ============================================================
> # 5. 路由
> # ============================================================
>
> @app.get("/health")
> def health():
>     return {
>         "status": "ok" if holder.agent is not None else "loading",
>         "model": f"{LOCAL_MODEL_PATH}/{SUBFOLDER}" if SUBFOLDER else LOCAL_MODEL_PATH,
>     }
>
>
> @app.post("/rerank", response_model=RerankResponse)
> async def rerank(req: RerankRequest):
>     if holder.agent is None:
>         raise HTTPException(503, "模型尚未加载完成")
>     if not req.documents:
>         return RerankResponse(results=[], latency_ms=0.0)
>
>     t0 = time.time()
>     docs = [{"id": d.id, "text": d.text} for d in req.documents]
>
>     # 同步推理丢到线程池，避免阻塞 event loop
>     loop = asyncio.get_running_loop()
>     try:
>         results = await asyncio.wait_for(
>             loop.run_in_executor(
>                 None,
>                 _rerank_sync,
>                 holder.agent, req.query, docs, req.top_k, req.batch_size,
>             ),
>             timeout=RERANK_TIMEOUT,
>         )
>     except asyncio.TimeoutError:
>         raise HTTPException(504, f"重排超时（>{RERANK_TIMEOUT}s）")
>
>     latency = (time.time() - t0) * 1000
>     return RerankResponse(
>         results=[RerankItem(**r) for r in results],
>         latency_ms=round(latency, 2),
>     )
>
>
> @app.post("/classify", response_model=ClassifyResponse)
> async def classify(req: ClassifyRequest):
>     if holder.agent is None:
>         raise HTTPException(503, "模型尚未加载完成")
>
>     t0 = time.time()
>     loop = asyncio.get_running_loop()
>     res = await loop.run_in_executor(
>         None, holder.agent.predict, req.state, req.questions
>     )
>     latency = (time.time() - t0) * 1000
>     return ClassifyResponse(answers=res["answers"], latency_ms=round(latency, 2))
>
>
> if __name__ == "__main__":
>     import uvicorn
>     uvicorn.run(
>         "server:app",
>         host="0.0.0.0",
>         port=8000,
>         workers=1,          # 重要：见下面说明
>     )
> ```

####  **kev**
（Jared Palmer 用 Devin 编写）基于 Qwen3.5 做 LoRA + pointer head 复刻 Jev 接口，镜像 System One API，官方 SDK 可直接指向本地服务，kev 训练只用交叉熵 + 温度缩放做校准，**没有复现 Jev 的 RLCD**（RL 校准决策那套），训练数据也明确声明"No Jev outputs were used for training"，所以它复现的是 Jev 的架构和 API 形态，训练目标函数是简化版。

| 模型           | 基座           | 定位                                                     |
| ------------ | ------------ | ------------------------------------------------------ |
| **Kev-0.8B** | Qwen3.5-0.8B | 轻量版，4GB 显存可微调，测试集 0.684                                |
| **Kev-4B**   | Qwen3.5-4B   | 推荐款，32GB Mac 可跑，测试集 0.837                              |
| **Kev-9B**   | Qwen3.5-9B   | 最准，测试集 0.852 / Brier 0.237，新源 dev 集 0.822 vs Jev 0.857 |

