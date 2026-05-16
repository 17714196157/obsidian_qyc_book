---
title: "vLLM Think方案"
source: "https://www.kimi.com/chat/19e30e71-c252-8b5a-8000-09fa70f94fe3?chat_enter_method=new_chat"
author:
published:
created: 2026-05-16
description: "Kimi K2.6 模型发布！全新建站功能可生成极具设计感的网站，支持轻量后端模块；Agent 集群全面升级，Office 文档一键转可复用技能。Claw 群组同步开启内测，探索多 Agent 协作的全新可能。"
tags:
  - "clippings"
---

Think-Deep,Not-Just-Long-Think@n实现.md

https://docs.vllm.ai/en/stable/examples/features/logits\_processor/#key-concepts 基于vllm的这个接口， 我能自己改一个 think@n的实现吗

基于 vLLM 的 logits processor 接口实现 Think@n 策略， **技术上可行，但架构上不匹配** 。核心问题在于：Think@n 需要 **跨多个候选序列的 DTR 比较和筛选** ，而 logits processor 是 **单步、单序列** 的局部操作。

## 为什么不直接用 logits processor 实现 Think@n

| Think@n 需求 | logits processor 能力 | 匹配度 |
| --- | --- | --- |
| 生成 n 个独立前缀 | 只能修改当前步 logits，不能分叉生成 | ❌ |
| 计算每个前缀的 DTR（需 hidden states） | 只能访问 logits，无 hidden states | ❌ |
| 按 DTR 排序并筛选 top-η% | 无跨请求比较机制 | ❌ |
| 对选中候选完整生成 | 无法控制生成流程的启停 | ❌ |
| 多数投票 | 无聚合逻辑 | ❌ |

logits processor 的设计定位是 **在单个生成序列的每一步修改输出分布** （如温度调整、token 屏蔽、结构化约束），而 Think@n 是一个 **多阶段、多序列的元策略** 。

---

## 在 vLLM 中实现 Think@n 的正确方式

### 方案一：客户端层实现（推荐）

在调用 vLLM 的 **客户端代码** 中封装 Think@n 逻辑，vLLM 只负责高效推理：

```python
import asyncio
import numpy as np
from collections import Counter
from vllm import LLM, SamplingParams

class VLLMThinkAtN:
    """
    基于 vLLM 离线推理 API 的 Think@n 实现
    """
    
    def __init__(self, model_name: str, **vllm_kwargs):
        self.llm = LLM(model=model_name, **vllm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
    
    def think_at_n(
        self,
        prompt: str,
        n: int = 8,              # 候选数
        prefix_len: int = 50,     # 前缀长度
        top_ratio: float = 0.5,   # 筛选比例
        max_new_tokens: int = 512,
        temperature: float = 1.0,
    ):
        # ========== Phase 1: 生成 n 个前缀 ==========
        prefix_params = SamplingParams(
            temperature=temperature,
            max_tokens=prefix_len,
            # 需要获取 hidden_states，但 vLLM 默认不暴露
            # 解决方案见下方
        )
        
        # 批量生成 n 个前缀
        prompts = [prompt] * n
        prefix_outputs = self.llm.generate(prompts, prefix_params)
        
        # ========== Phase 2: 计算 DTR ==========
        # ⚠️ 关键难点：vLLM 的 generate() 默认不返回 hidden_states
        dtr_scores = []
        for output in prefix_outputs:
            # 需要自定义方式获取 hidden_states 计算 DTR
            dtr = self._compute_dtr_from_output(output)
            dtr_scores.append(dtr)
        
        # ========== Phase 3: 筛选 Top-η% ==========
        ranked = np.argsort(dtr_scores)[::-1]
        top_k = max(1, int(n * top_ratio))
        selected = ranked[:top_k]
        
        # ========== Phase 4: 完整生成 ==========
        final_params = SamplingParams(
            temperature=0.0,  # 确定性生成
            max_tokens=max_new_tokens,
        )
        
        # 构建前缀提示继续生成
        selected_prompts = []
        for idx in selected:
            prefix_text = prefix_outputs[idx].outputs[0].text
            selected_prompts.append(prompt + prefix_text)
        
        final_outputs = self.llm.generate(selected_prompts, final_params)
        
        # ========== Phase 5: 多数投票 ==========
        answers = [o.outputs[0].text for o in final_outputs]
        final_answer = Counter(answers).most_common(1)[0][0]
        
        return {
            "answer": final_answer,
            "candidates": answers,
            "dtr_scores": dtr_scores,
        }
    
    def _compute_dtr_from_output(self, output):
        """从 vLLM 输出计算 DTR - 需要 hidden_states"""
        # vLLM 默认不返回 hidden_states，需要修改
        raise NotImplementedError("需要启用 hidden_states 输出")
```

### 方案二：vLLM 服务端扩展（生产环境）

如果要在 vLLM 的\*\* serving 模式\*\*（OpenAI API 兼容）中支持 Think@n，需要：

1. **新增自定义 API 端点** （非标准 `/v1/completions` ）
2. **在服务端实现多阶段调度逻辑**

```python
# 伪代码：vLLM serving 扩展
from vllm.entrypoints.openai.api_server import router
from fastapi import Request

@router.post("/v1/think_at_n")
async def think_at_n(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    n = body.get("n", 8)
    
    # 内部调用引擎多次生成、计算 DTR、筛选、再生成
    # 这需要访问引擎内部状态，目前 vLLM 没有直接暴露
    ...
```

---

## 关键障碍：vLLM 不暴露 hidden\_states

Think@n 的核心是 DTR 计算，DTR 需要 **逐层 hidden states** 。但 vLLM 的设计高度优化了推理性能，默认 **不保留中间层 hidden states** ：

- vLLM 使用 PagedAttention 和连续批处理，hidden states 在 GPU 上流动后即释放
- `output_hidden_states=True` 在 vLLM 的 `generate()` 中 **不被支持** （与 HuggingFace 不同）

### 解决方案对比

| 方案 | 可行性 | 性能影响 | 实现复杂度 |
| --- | --- | --- | --- |
| **修改 vLLM 源码** 添加 `output_hidden_states` 支持 | 可行 | 显存占用大幅增加 | 高（需改 C++/CUDA 层） |
| **用 HuggingFace 做 DTR，vLLM 做生成** | 可行 | 需维护两套模型 | 中 |
| **近似 DTR：用 logits 变化代替 hidden states** | 可行 | 无额外开销 | 低（可用 logits processor） |

---

## 方案三：用 logits processor 做"轻量版 Think@n"

如果一定要在 logits processor 接口内实现，可以做一个 **近似版** ，用 **logits 的熵/方差** 代替 DTR 作为"思考深度"的代理指标：

```python
import torch
import numpy as np
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate
from vllm.sampling_params import SamplingParams

class LightweightThinkAtNProcessor(LogitsProcessor):
    """
    轻量版 Think@n logits processor
    用 logits 分布的"犹豫程度"（熵）近似 DTR
    限制：只能单序列操作，无法实现真正的多候选筛选
    """
    
    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra = params.extra_args or {}
        if "think_at_n_mode" in extra:
            mode = extra["think_at_n_mode"]
            if mode not in ["measure", "generate"]:
                raise ValueError("think_at_n_mode must be 'measure' or 'generate'")
    
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.entropy_history = {}  # req_id -> list of entropy values
        self.req_modes = {}        # req_id -> "measure" or "generate"
    
    def is_argmax_invariant(self) -> bool:
        return True  # 我们不修改 argmax，只记录统计信息
    
    def update_state(self, batch_update):
        if not batch_update:
            return
        # 处理新增请求
        for idx, params, _, _ in batch_update.added:
            extra = params.extra_args or {}
            self.req_modes[idx] = extra.get("think_at_n_mode", "generate")
            self.entropy_history[idx] = []
        # 处理移除请求
        for idx in batch_update.removed:
            self.entropy_history.pop(idx, None)
            self.req_modes.pop(idx, None)
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算当前步 logits 的熵作为"思考深度"代理
        不修改 logits，只记录统计
        """
        # 计算每个请求的 logits 熵
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        
        # 记录到历史（后续可用于判断是否"深度思考"）
        for req_idx in range(logits.shape[0]):
            # 这里 req_idx 是 batch 内索引，实际需要用 batch_update 映射到真实 req_id
            pass
        
        return logits  # 不修改 logits
```

**这个方案的局限** ：

- 只能观察单序列的"犹豫程度"，无法像 Think@n 那样 **跨候选比较**
- 无法基于 DTR 排序后 **重新选择生成路径**
- 熵高不一定等于"深度思考"，可能只是模型不确定

---

## 推荐实现路径

| 场景 | 推荐方案 |
| --- | --- |
| **离线实验/研究** | 直接用上传的 HuggingFace 实现，不用 vLLM |
| **生产环境需要 Think@n** | **客户端实现** ：vLLM 只负责高效生成，Think@n 逻辑在客户端编排 |
| **必须在 vLLM 内部实现** | 需要 fork vLLM 源码，添加 `output_hidden_states` 支持，然后在内核中集成 DTR 计算 |

### 客户端实现的完整架构

```markdown
┌─────────────────┐
│   Client App    │
│  (Think@n 逻辑)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌───────┐
│ vLLM  │  │ vLLM  │  ← 两次调用，或两个独立实例
│ 实例1  │  │ 实例2  │    实例1: 采样生成前缀 (temperature>0)
└───────┘  └───────┘    实例2: 确定性完整生成 (temperature=0)
    │
    ▼
┌───────────────┐
│  DTR 计算模块  │  ← 需要 hidden_states，可用小模型在 HF 上计算
│ (HuggingFace) │
└───────────────┘
```

**核心洞察** ：Think@n 是 **策略层** （meta-strategy），logits processor 是 **采样层** （sampling primitive）。策略层应该在客户端或更高层实现，采样层保持精简高效。

我就是要用 配合 vllm 推理 实现 DTR 作为"思考深度"的代理指标 的 Think@n 方案 ，给我想想办法


明白了！你想要的是 **基于 vLLM 的高性能推理，但用 DTR（或近似 DTR）作为筛选指标** 的 Think@n 方案。

核心思路： **vLLM 负责高效生成，DTR 计算外挂，通过两次调用来实现筛选** 。这里给你一个完整的工程实现方案。

---

## 架构设计

```markdown
┌─────────────────────────────────────────┐
│           Think@n Controller              │
│         (客户端 / 服务层)                 │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ vLLM   │  │ vLLM   │  │ DTR    │
   │ Prefix │  │ Final  │  │ Engine │
   │ Gen    │  │ Gen    │  │ (HF)   │
   │ n=8    │  │ top_k=4│  │        │
   └────────┘  └────────┘  └────────┘
        │           │           │
        └───────────┴───────────┘
                    ▼
           ┌──────────────┐
           │ Majority Vote│
           └──────────────┘
```

---

## 完整实现代码

```python
"""
Think@n with vLLM + DTR Approximation
高性能版本：vLLM 生成 + 轻量 DTR 计算
"""

import asyncio
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. DTR 计算器（轻量版，基于 logits 序列）
# ============================================================

@dataclass
class DTRConfig:
    settling_threshold: float = 0.5
    depth_fraction: float = 0.85
    min_settling_layers: int = 3

class LogitsBasedDTR:
    """
    基于 logits 序列的轻量 DTR 计算器
    不需要 hidden_states，只需要每层的 logits 输出
    """
    
    def __init__(self, model: AutoModelForCausalLM, config: Optional[DTRConfig] = None):
        self.model = model
        self.model.eval()
        self.config = config or DTRConfig()
        
        # 获取 lm_head
        if hasattr(model, 'lm_head'):
            self.lm_head = model.lm_head
        elif hasattr(model, 'embed_out'):
            self.lm_head = model.embed_out
        else:
            self.lm_head = model.get_input_embeddings()
        
        self.num_layers = model.config.num_hidden_layers
    
    def compute_dtr_from_logits_sequence(
        self, 
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> float:
        """
        通过逐层前向传播计算 DTR
        对 prefix 中的每个 token，计算其逐层 logits 的 JSD
        """
        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        seq_len = generated_ids.shape[1]
        
        if seq_len == 0:
            return 0.0
        
        deep_thinking_count = 0
        
        with torch.no_grad():
            # 获取所有层的 hidden_states
            outputs = self.model(
                full_ids,
                output_hidden_states=True,
                use_cache=False
            )
            
            hidden_states = outputs.hidden_states  # tuple of (L+1) tensors
            num_layers = len(hidden_states) - 1
            
            # 对 prefix 中的每个生成 token 计算 DTR
            for pos in range(input_ids.shape[1], full_ids.shape[1]):
                # 获取该位置所有层的 hidden state
                token_hs = torch.stack([hs[0, pos, :] for hs in hidden_states], dim=0)
                
                # 逐层投影到词表
                final_proj = self._project(token_hs[-1])
                
                jsd_values = []
                for l in range(1, num_layers + 1):
                    layer_proj = self._project(token_hs[l])
                    jsd = self._jsd(layer_proj, final_proj)
                    jsd_values.append(jsd.item())
                
                # 计算沉降深度
                settling_depth = self._settling_depth(
                    jsd_values, 
                    self.config.settling_threshold,
                    self.config.min_settling_layers
                )
                
                # 判定是否深度思考
                deep_threshold = int((1 - self.config.depth_fraction) * num_layers)
                if settling_depth >= deep_threshold:
                    deep_thinking_count += 1
        
        return deep_thinking_count / seq_len
    
    def _project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_state)
        return F.softmax(logits, dim=-1)
    
    def _jsd(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        m = 0.5 * (p + q + 1e-12)
        kl_pm = F.kl_div(m.log(), p + 1e-12, reduction='sum')
        kl_qm = F.kl_div(m.log(), q + 1e-12, reduction='sum')
        return 0.5 * (kl_pm + kl_qm)
    
    def _settling_depth(self, jsd_values: List[float], threshold: float, min_consecutive: int) -> int:
        consecutive = 0
        for i, jsd in enumerate(jsd_values):
            if jsd <= threshold:
                consecutive += 1
                if consecutive >= min_consecutive:
                    return i - min_consecutive + 1
            else:
                consecutive = 0
        return len(jsd_values) - 1

# ============================================================
# 2. 超轻量 DTR：仅基于输出 logits 的熵/方差
# ============================================================

class EntropyBasedDTR:
    """
    超轻量 DTR 近似：不需要额外模型，只分析 vLLM 输出的 logits
    用 logits 分布的"犹豫程度"（熵）+ 跨步变化率 作为思考深度代理
    """
    
    def __init__(self, config: Optional[DTRConfig] = None):
        self.config = config or DTRConfig()
    
    def compute_from_logits_list(self, logits_list: List[torch.Tensor]) -> float:
        """
        从 logits 列表计算近似 DTR
        
        Args:
            logits_list: 每步的 logits tensor，shape (vocab_size,)
        """
        if not logits_list:
            return 0.0
        
        entropies = []
        entropy_changes = []
        
        for i, logits in enumerate(logits_list):
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
            entropies.append(entropy)
            
            if i > 0:
                change = abs(entropy - entropies[i-1])
                entropy_changes.append(change)
        
        # 高熵 + 高变化率 = 深度思考
        avg_entropy = np.mean(entropies)
        avg_change = np.mean(entropy_changes) if entropy_changes else 0
        
        # 归一化到 [0, 1] 作为 DTR 代理
        # 假设最大熵约为 log(vocab_size) ~ 15 (对于 32k vocab)
        normalized_entropy = min(avg_entropy / 15.0, 1.0)
        normalized_change = min(avg_change / 5.0, 1.0)
        
        # 加权组合
        dtr_proxy = 0.6 * normalized_entropy + 0.4 * normalized_change
        
        return float(dtr_proxy)

# ============================================================
# 3. vLLM Think@n 核心实现
# ============================================================

@dataclass
class ThinkAtNConfig:
    num_candidates: int = 8          # n
    prefix_length: int = 50          # 前缀长度
    top_ratio: float = 0.5           # η
    max_new_tokens: int = 1024       # 完整生成长度
    temperature: float = 1.0         # 前缀采样温度
    dtr_mode: str = "hybrid"         # "full"(HF模型), "entropy"(纯logits), "hybrid"(组合)
    dtr_model_path: Optional[str] = None  # 用于 full DTR 的小模型路径

class VLLMThinkAtN:
    """
    vLLM 原生 Think@n 实现
    支持三种 DTR 计算模式
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[ThinkAtNConfig] = None,
        vllm_kwargs: Optional[Dict] = None,
    ):
        self.config = config or ThinkAtNConfig()
        self.vllm_kwargs = vllm_kwargs or {}
        
        # 初始化 vLLM 引擎（主推理引擎）
        print(f"🚀 初始化 vLLM 引擎: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=self.vllm_kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.vllm_kwargs.get("gpu_memory_utilization", 0.9),
            dtype=self.vllm_kwargs.get("dtype", "auto"),
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # 初始化 DTR 计算组件
        self.dtr_full = None
        self.dtr_entropy = EntropyBasedDTR()
        
        if self.config.dtr_mode in ["full", "hybrid"]:
            if self.config.dtr_model_path:
                print(f"📦 加载 DTR 计算模型: {self.config.dtr_model_path}")
                dtr_model = AutoModelForCausalLM.from_pretrained(
                    self.config.dtr_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.dtr_full = LogitsBasedDTR(dtr_model)
            else:
                # 复用 vLLM 的模型做 DTR（显存开销大，不推荐）
                print("⚠️ 复用主模型计算 DTR，显存占用较高")
                # 需要从 vLLM 内部提取模型，较复杂，暂不实现
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def generate(self, prompt: str) -> Dict:
        """执行完整的 Think@n 流程"""
        n = self.config.num_candidates
        prefix_len = self.config.prefix_length
        top_k = max(1, int(n * self.config.top_ratio))
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # ========== Phase 1: 并行生成 n 个前缀 ==========
        print(f"\n🔍 Phase 1: 生成 {n} 个前缀 (各 {prefix_len} tokens)")
        
        prefix_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=prefix_len,
            # vLLM 0.6.0+ 支持获取 logits
            output_logits=True,  # 获取每步 logits
            prompt_logprobs=0,   # 获取 prompt 的 logprobs
        )
        
        # 批量生成
        outputs = self.llm.generate([prompt] * n, prefix_params)
        
        # 提取前缀文本和 logits
        prefixes = []
        logits_sequences = []  # 每个候选的 logits 序列
        
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            prefixes.append(text)
            
            # 提取 logits（如果 vLLM 支持）
            if hasattr(output.outputs[0], 'logits') and output.outputs[0].logits:
                logits_sequences.append(output.outputs[0].logits)
            else:
                logits_sequences.append(None)
            
            print(f"  候选 {i+1}: {text[:60]}...")
        
        # ========== Phase 2: 计算 DTR ==========
        print(f"\n📊 Phase 2: 计算 DTR (模式: {self.config.dtr_mode})")
        
        dtr_scores = []
        
        for i in range(n):
            if self.config.dtr_mode == "entropy" and logits_sequences[i] is not None:
                # 纯熵模式：直接用 vLLM 输出的 logits
                dtr = self.dtr_entropy.compute_from_logits_list(logits_sequences[i])
                
            elif self.config.dtr_mode in ["full", "hybrid"] and self.dtr_full is not None:
                # 完整 DTR 模式：用 HF 模型计算
                prefix_ids = self.tokenizer.encode(prefixes[i], return_tensors="pt")
                dtr = self.dtr_full.compute_dtr_from_logits_sequence(input_ids, prefix_ids)
                
            else:
                # 回退：用熵近似
                if logits_sequences[i]:
                    dtr = self.dtr_entropy.compute_from_logits_list(logits_sequences[i])
                else:
                    # 无法获取 logits，用随机值（退化情况）
                    dtr = np.random.random()
            
            dtr_scores.append(dtr)
            print(f"  候选 {i+1}: DTR = {dtr:.4f}")
        
        # ========== Phase 3: 筛选 Top-η% ==========
        ranked_indices = np.argsort(dtr_scores)[::-1]
        selected_indices = ranked_indices[:top_k]
        
        print(f"\n✅ Phase 3: 选择 Top-{top_k}")
        print(f"   排序: {[(i, f'{dtr_scores[i]:.4f}') for i in ranked_indices]}")
        print(f"   选中: {selected_indices.tolist()}")
        
        # ========== Phase 4: 完整生成 ==========
        print(f"\n🚀 Phase 4: 对选中候选完整生成")
        
        final_params = SamplingParams(
            temperature=0.0,  # 确定性
            max_tokens=self.config.max_new_tokens,
        )
        
        # 构建完整 prompt：原 prompt + 前缀
        final_prompts = []
        for idx in selected_indices:
            full_prompt = prompt + prefixes[idx]
            final_prompts.append(full_prompt)
        
        final_outputs = self.llm.generate(final_prompts, final_params)
        
        answers = []
        total_tokens = 0
        for i, output in enumerate(final_outputs):
            text = output.outputs[0].text
            answers.append(text)
            total_tokens += len(output.outputs[0].token_ids)
            print(f"  候选 {selected_indices[i]+1}: {text[:80]}...")
        
        # ========== Phase 5: 多数投票 ==========
        final_answer = self._majority_vote(answers)
        
        return {
            "answer": final_answer,
            "candidates": answers,
            "dtr_scores": dtr_scores,
            "prefixes": prefixes,
            "selected_indices": selected_indices.tolist(),
            "cost_tokens": total_tokens,
        }
    
    def _majority_vote(self, answers: List[str]) -> str:
        """多数投票，支持数值答案的近似匹配"""
        normalized = [a.strip().lower() for a in answers]
        counter = Counter(normalized)
        return counter.most_common(1)[0][0]
    
    def batch_generate(self, prompts: List[str]) -> List[Dict]:
        """批量处理多个 prompt"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt)
            results.append(result)
        return results

# ============================================================
# 4. 异步版本（用于服务化部署）
# ============================================================

class AsyncVLLMThinkAtN:
    """异步版本的 Think@n，适合高并发场景"""
    
    def __init__(self, sync_impl: VLLMThinkAtN):
        self.sync = sync_impl
        self.loop = asyncio.get_event_loop()
    
    async def generate(self, prompt: str) -> Dict:
        """异步包装"""
        return await self.loop.run_in_executor(
            None,  # 默认 executor
            self.sync.generate,
            prompt
        )

# ============================================================
# 5. 使用示例
# ============================================================

def main():
    """演示 Think@n 使用"""
    
    # 配置
    config = ThinkAtNConfig(
        num_candidates=8,
        prefix_length=50,
        top_ratio=0.5,
        max_new_tokens=512,
        temperature=1.0,
        dtr_mode="entropy",  # 先用纯熵模式测试
    )
    
    # 初始化（使用 Qwen2.5-7B 作为示例）
    think_at_n = VLLMThinkAtN(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        config=config,
        vllm_kwargs={
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
        }
    )
    
    # 测试问题
    prompts = [
        """Solve the following math problem step by step:

A right triangle has legs of length 3 and 4. What is the area of the triangle?
Provide your final answer in \\boxed{}.""",
        
        """What is the capital of France? Explain your reasoning.""",
    ]
    
    for prompt in prompts:
        print("\n" + "="*70)
        print(f"❓ Prompt: {prompt[:100]}...")
        
        result = think_at_n.generate(prompt)
        
        print(f"\n🎯 最终答案: {result['answer']}")
        print(f"💰 Token 消耗: {result['cost_tokens']}")
        print(f"📊 DTR 分布: {[f'{s:.3f}' for s in result['dtr_scores']]}")

if __name__ == "__main__":
    main()
```

---

## 关键技术点说明

### 1\. vLLM 获取 logits 的方式

vLLM 0.6.0+ 支持在 `SamplingParams` 中设置：

- `output_logits=True` ：返回每步的 logits（注意：这会显著增加返回数据量）
- `prompt_logprobs=0` ：获取 prompt 部分的 logprobs

如果 vLLM 版本不支持 `output_logits` ，可以：

1. **升级 vLLM** ： `pip install -U vllm`
2. **用 logprobs 反推** ：vLLM 默认返回 `logprobs` ，可以通过 `logprobs` 还原近似 logits

### 2\. 三种 DTR 模式选择

| 模式 | 计算位置 | 显存开销 | 精度 | 适用场景 |
| --- | --- | --- | --- | --- |
| `entropy` | 客户端 CPU | 无额外开销 | 中 | **推荐** ，生产环境 |
| `full` | 独立 HF 模型 GPU | 需额外模型 | 高 | 研究/精度敏感 |
| `hybrid` | 组合计算 | 中等 | 高 | 平衡方案 |

### 3\. 性能优化建议

```python
# 1. 前缀生成并行化（vLLM 自动 batch）
prefix_params = SamplingParams(
    temperature=1.0,
    max_tokens=50,
    # 使用 speculative decoding 加速
    speculative_model="[ smaller draft model ]",  
)

# 2. 完整生成使用确定性解码 + KV Cache 复用
# vLLM 会自动复用 prefix 的 KV Cache！

# 3. DTR 计算异步化
import asyncio
async def compute_dtr_batch(logits_list):
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, dtr_entropy.compute, logits)
        for logits in logits_list
    ]
    return await asyncio.gather(*tasks)
```

### 4\. 显存优化：用小模型做 DTR

```python
# 用 1B 级别小模型做 DTR 计算，主模型用 7B/70B
config = ThinkAtNConfig(
    dtr_mode="full",
    dtr_model_path="Qwen/Qwen2.5-1.5B-Instruct",  # 小模型专门算 DTR
)

# 主推理模型
think_at_n = VLLMThinkAtN(
    model_path="Qwen/Qwen2.5-72B-Instruct",  # 大模型做生成
    config=config,
)
```

---

## 部署架构（生产环境）

```markdown
┌─────────────────────────────────────────┐
│           API Gateway                   │
│         (FastAPI/Traefik)               │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ vLLM   │  │ vLLM   │  │ DTR    │
   │ Prefix │  │ Final  │  │ Service│
   │ (GPU1) │  │ (GPU2) │  │ (GPU3) │
   └────────┘  └────────┘  └────────┘
        │           │           │
        └───────────┴───────────┘
                    ▼
           ┌──────────────┐
           │ Redis Cache  │  ← 缓存常见 prompt 的 DTR
           │ (前缀复用)    │
           └──────────────┘
```

这个方案的核心优势：

1. **vLLM 负责它擅长的** ：高速并行生成
2. **DTR 计算解耦** ：可以独立扩展、降级
3. **两次调用模型** ：Phase 1 和 Phase 4 可以共享同一个 vLLM 实例（通过不同的 `SamplingParams` ）
4. **KV Cache 自动复用** ：vLLM 会自动缓存 prefix 的 KV，Phase 4 的完整生成可以直接复用

尽管问，带图也行

K2.6 快速