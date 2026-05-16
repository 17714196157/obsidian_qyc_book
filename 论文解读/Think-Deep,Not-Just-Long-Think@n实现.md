---
title: Think-Deep, Not-Just-Long —— Think@n 策略实现
source: https://arxiv.org/pdf/2602.13517
created: 2026-05-15
tags:
  - paper/implementation
  - LLM/optimization
  - DTR
  - think-at-n
---
vLLM Server 模式下实现 Think@n，核心思路是**客户端编排 + 服务端高效推理**。服务端保持纯 vLLM，所有策略逻辑放在客户端或中间层。
```
┌─────────────────────────────────────────┐
│         Client / Orchestrator           │
│    (Think@n 策略逻辑：筛选、投票)          │
│         Python / Go / Node.js           │
└─────────────────────────────────────────┘
                    │
                    ▼ HTTP API
┌─────────────────────────────────────────┐
│           vLLM OpenAI Server              │
│    ┌─────────────────────────────┐      │
│    │  /v1/completions           │      │
│    │  /v1/chat/completions      │      │
│    │  (标准 OpenAI API)         │      │
│    └─────────────────────────────┘      │
│              GPU Cluster                │
└─────────────────────────────────────────┘
```

### 关键优化点
##### 1. **利用 vLLM 的 `n` 参数（核心优化）**
vLLM 的 `/v1/chat/completions` 支持 `n > 1`，**单次请求内部并行生成 n 个候选**：
这比发 8 次独立请求**快得多**，因为：
- 共享 prompt 的 KV Cache
- 同一 batch 内并行调度
- 减少网络往返
```PYTHON
# 单次请求生成 8 个前缀，vLLM 内部并行调度
prefix_resp = await client.chat_completion(
    messages=messages,
    max_tokens=50,
    n=8,  # vLLM 内部并行生成 8 个！
    temperature=1.0,
)
```

##### DTR 计算零开销
```python
# DTR 完全基于 vLLM 返回的 logprobs，纯客户端 CPU 计算
# 不需要额外模型，不需要 GPU 显存
dtr = LogitsEntropyDTR.compute_from_logprobs(logprobs)
```
计算复杂度：O(n × prefix_len × vocab_size)，在 CPU 上毫秒级完成。

### 完整代码实现
```python
"""
Think@n with vLLM Server — 完整客户端实现
兼容 vLLM OpenAI API，支持 Qwen2.5-14B-Instruct

vLLM logprobs 参数说明:
    - logprobs: bool 类型，True/False 是否启用
    - top_logprobs: int 类型，返回 top N 个 (0-20)
"""

import asyncio
import aiohttp
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from collections import Counter
import time
import json


# ============================================================
# 1. 配置定义
# ============================================================

@dataclass
class ServerConfig:
    base_url: str = "http://192.168.0.180:8103/v1"
    api_key: str = "dummy"
    timeout: float = 120.0
    max_concurrent: int = 16


@dataclass
class ThinkAtNConfig:
    num_candidates: int = 8          # n: 初始采样数
    prefix_length: int = 50          # 前缀长度
    top_ratio: float = 0.5           # η: 筛选比例
    max_new_tokens: int = 1024       # 完整生成长度
    temperature: float = 1.0         # 前缀采样温度
    top_p: float = 0.95
    dtr_mode: str = "logits_entropy"
    majority_vote: str = "exact"      # exact | fuzzy


# ============================================================
# 2. DTR 计算器（兼容 vLLM 列表格式）
# ============================================================

class LogitsEntropyDTR:
    """
    基于 vLLM 返回的 logprobs 计算 DTR 代理指标
    vLLM 格式: top_logprobs = [{"token": "x", "logprob": -0.5}, ...]
    """

    @staticmethod
    def compute_from_logprobs(logprobs_list: List[Any]) -> float:
        """
        从每步的 top_logprobs 列表计算"思考深度"代理

        Args:
            logprobs_list: 每步的 top_logprobs 数据
                vLLM 格式: [{"token": "To", "logprob": -0.01}, ...]
        """
        if not logprobs_list:
            return 0.0

        entropies = []
        top1_confidences = []
        entropy_changes = []

        for i, step_logprobs in enumerate(logprobs_list):
            if not step_logprobs:
                continue

            # vLLM 格式处理: step_logprobs 是列表
            logprob_values = []
            if isinstance(step_logprobs, list):
                for item in step_logprobs:
                    if isinstance(item, dict) and "logprob" in item:
                        logprob_values.append(item["logprob"])

            elif isinstance(step_logprobs, dict):
                logprob_values = list(step_logprobs.values())

            if not logprob_values:
                continue

            # 转换为概率
            log_probs = np.array(logprob_values, dtype=np.float64)

            # 数值稳定：减去最大值防止 exp 溢出
            max_logprob = np.max(log_probs)
            log_probs_stable = log_probs - max_logprob
            probs = np.exp(log_probs_stable)
            probs = probs / (probs.sum() + 1e-12)

            # 计算熵
            entropy = -np.sum(probs * (log_probs + 1e-12))
            entropies.append(entropy)

            # Top-1 置信度
            top1_conf = np.max(probs)
            top1_confidences.append(top1_conf)

            # 熵变化率
            if i > 0:
                change = abs(entropy - entropies[i - 1])
                entropy_changes.append(change)

        if not entropies:
            return 0.0

        # 深度思考特征
        avg_entropy = np.mean(entropies)
        avg_top1_conf = np.mean(top1_confidences)
        avg_change = np.mean(entropy_changes) if entropy_changes else 0

        # 归一化
        norm_entropy = min(avg_entropy / 8.0, 1.0)
        norm_uncertainty = 1.0 - avg_top1_conf
        norm_change = min(avg_change / 3.0, 1.0)

        # 加权 DTR 代理
        dtr = 0.4 * norm_entropy + 0.4 * norm_uncertainty + 0.2 * norm_change

        return float(dtr)

    @staticmethod
    def compute_from_completion(completion: Dict) -> float:
        """
        从 vLLM / OpenAI API 的 completion 响应提取 logprobs 计算 DTR
        """
        logprobs_data = completion.get("logprobs")
        if not logprobs_data:
            return 0.0

        # vLLM 格式: logprobs.content 是每步的 token 信息列表
        content = logprobs_data.get("content", [])
        if not content:
            return 0.0

        # 提取每步的 top_logprobs
        steps = []
        for step in content:
            if isinstance(step, dict):
                top_logprobs = step.get("top_logprobs")
                if top_logprobs:
                    steps.append(top_logprobs)

        return LogitsEntropyDTR.compute_from_logprobs(steps)


# ============================================================
# 3. vLLM HTTP 客户端
# ============================================================

class VLLMClient:
    """vLLM Server 的异步 HTTP 客户端"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _request(self, endpoint: str, payload: Dict) -> Dict:
        """发送 HTTP 请求"""
        url = f"{self.config.base_url}/{endpoint}"

        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text}")
                    return await resp.json()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: bool = False,      # vLLM: bool 类型！
        top_logprobs: int = 0,       # vLLM: int 类型，0-20
        n: int = 1,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict] = None,
    ) -> Dict:
        """
        调用 /v1/chat/completions
        vLLM 支持 n > 1 并行生成
        """
        payload = {
            "model": "qwen2.5-14b-instruct",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
        }

        # vLLM: logprobs 是 bool，top_logprobs 是 int (0-20)
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = min(max(top_logprobs, 0), 20)

        if stop:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        return await self._request("chat/completions", payload)


# ============================================================
# 4. Think@n 核心实现
# ============================================================

class ThinkAtNServer:
    """
    vLLM Server 模式的 Think@n 实现
    所有策略逻辑在客户端，服务端保持纯净 vLLM
    """

    def __init__(
        self,
        server_config: Optional[ServerConfig] = None,
        think_config: Optional[ThinkAtNConfig] = None,
    ):
        self.server_config = server_config or ServerConfig()
        self.think_config = think_config or ThinkAtNConfig()
        self.client = VLLMClient(self.server_config)
        self.dtr_calc = LogitsEntropyDTR()

    async def think_at_n(self, prompt: str, system: Optional[str] = None) -> Dict:
        """
        执行完整的 Think@n 流程

        流程：
        1. 单次请求生成 n 个前缀（利用 vLLM 的 n 参数并行）
        2. 计算每个前缀的 DTR
        3. 筛选 Top-η%
        4. 对选中候选完整生成（并行请求）
        5. 多数投票
        """
        n = self.think_config.num_candidates
        prefix_len = self.think_config.prefix_length
        top_k = max(1, int(n * self.think_config.top_ratio))

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # ========== Phase 1: 生成 n 个前缀（单次请求，vLLM 内部并行）==========
        print(f"\n🔍 Phase 1: 生成 {n} 个前缀 (利用 vLLM n={n} 并行)")
        start_time = time.time()

        prefix_resp = await self.client.chat_completion(
            messages=messages,
            max_tokens=prefix_len,
            temperature=self.think_config.temperature,
            top_p=self.think_config.top_p,
            logprobs=True,          # bool: 启用 logprobs
            top_logprobs=5,         # int: 返回 top 5 个
            n=n,                    # vLLM 内部并行生成 n 个候选
        )

        prefix_time = time.time() - start_time

        # 提取 n 个候选
        choices = prefix_resp.get("choices", [])
        if len(choices) < n:
            print(f"⚠️ 只获得 {len(choices)}/{n} 个候选")

        prefixes = []
        dtr_scores = []

        for i, choice in enumerate(choices):
            message = choice.get("message", {})
            text = message.get("content", "")
            prefixes.append(text)

            # 计算 DTR
            dtr = self.dtr_calc.compute_from_completion(choice)
            dtr_scores.append(dtr)

            print(f"  候选 {i+1}: DTR={dtr:.4f} | {text[:50]}...")

        print(f"   耗时: {prefix_time:.2f}s")

        # ========== Phase 2: DTR 排序，选 Top-k ==========
        ranked_indices = np.argsort(dtr_scores)[::-1]
        selected_indices = ranked_indices[:top_k]

        print(f"\n📊 Phase 2: DTR 排序")
        print(f"   全部: {[f'{dtr_scores[i]:.3f}' for i in ranked_indices]}")
        print(f"   选中 Top-{top_k}: {selected_indices.tolist()}")

        # ========== Phase 3: 完整生成（并行请求）==========
        print(f"\n🚀 Phase 3: 对 Top-{top_k} 完整生成")

        selected_prompts = []
        for idx in selected_indices:
            full_prompt = prompt + "\n\n" + prefixes[idx]
            selected_prompts.append(full_prompt)

        final_tasks = [
            self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system or "You are a helpful assistant."},
                    {"role": "user", "content": p},
                ],
                max_tokens=self.think_config.max_new_tokens,
                temperature=0.0,    # 确定性生成
                logprobs=False,     # 不需要 logprobs
                n=1,
            )
            for p in selected_prompts
        ]

        start_time = time.time()
        final_responses = await asyncio.gather(*final_tasks)
        final_time = time.time() - start_time

        answers = []
        total_tokens = 0
        for i, resp in enumerate(final_responses):
            text = resp["choices"][0]["message"]["content"]
            answers.append(text)
            total_tokens += resp.get("usage", {}).get("completion_tokens", 0)
            print(f"  候选 {selected_indices[i]+1}: {text[:80]}...")

        print(f"   耗时: {final_time:.2f}s")

        # ========== Phase 4: 多数投票 ==========
        final_answer = self._majority_vote(answers)

        total_time = prefix_time + final_time
        prefix_tokens = prefix_resp.get("usage", {}).get("total_tokens", 0)
        final_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in final_responses)
        cost_tokens = prefix_tokens + final_tokens

        return {
            "answer": final_answer,
            "candidates": answers,
            "dtr_scores": [float(s) for s in dtr_scores],
            "prefixes": prefixes,
            "selected_indices": selected_indices.tolist(),
            "prefix_time": prefix_time,
            "final_time": final_time,
            "total_time": total_time,
            "cost_tokens": cost_tokens,
        }

    def _majority_vote(self, answers: List[str]) -> str:
        """多数投票"""
        if self.think_config.majority_vote == "exact":
            normalized = [a.strip() for a in answers]
            counter = Counter(normalized)
            return counter.most_common(1)[0][0]

        elif self.think_config.majority_vote == "fuzzy":
            return self._fuzzy_vote(answers)

        else:
            return answers[0] if answers else ""

    def _fuzzy_vote(self, answers: List[str]) -> str:
        """模糊投票：提取结构化答案"""
        import re

        extracted = []
        for ans in answers:
            matches = re.findall(r'\\boxed\{([^}]+)\}', ans)
            if matches:
                extracted.append(matches[-1].strip())
            else:
                lines = ans.strip().split("\n")
                last_line = lines[-1]
                nums = re.findall(r'-?\d+\.?\d*', last_line)
                if nums:
                    extracted.append(nums[-1])
                else:
                    extracted.append(ans.strip()[:100])

        counter = Counter(extracted)
        winner = counter.most_common(1)[0][0]

        for ans, ext in zip(answers, extracted):
            if ext == winner:
                return ans

        return answers[0]

    async def batch_think_at_n(
        self,
        prompts: List[str],
        system: Optional[str] = None
    ) -> List[Dict]:
        """批量处理"""
        tasks = [self.think_at_n(p, system) for p in prompts]
        return await asyncio.gather(*tasks)


# ============================================================
# 5. 使用示例
# ============================================================

async def demo():
    """演示"""

    server_config = ServerConfig(
        base_url="http://192.168.0.180:8103/v1",
        max_concurrent=8,
    )

    think_config = ThinkAtNConfig(
        num_candidates=8,
        prefix_length=50,
        top_ratio=0.5,
        max_new_tokens=512,
        temperature=1.0,
        dtr_mode="logits_entropy",
        majority_vote="fuzzy",
    )

    think_at_n = ThinkAtNServer(server_config, think_config)

    prompt = """Solve the following math problem step by step:

A right triangle has legs of length 3 and 4. What is the area of the triangle?
Provide your final answer in \boxed{}.
"""

    print("=" * 70)
    print("Think@n with vLLM Server")
    print(f"Server: {server_config.base_url}")
    print("=" * 70)

    result = await think_at_n.think_at_n(prompt)

    print("\n" + "=" * 70)
    print(f"🎯 最终答案:\n{result['answer']}")
    print(f"\n⏱️  总耗时: {result['total_time']:.2f}s")
    print(f"💰 Token 消耗: {result['cost_tokens']}")
    print(f"📊 DTR 分数: {[f'{s:.3f}' for s in result['dtr_scores']]}")
    print(f"✅ 选中索引: {result['selected_indices']}")


if __name__ == "__main__":
    asyncio.run(demo())

```