---
title: vLLM 与 LiteLLM 实现 Guardrail 和负载分发
source: https://mp.weixin.qq.com/s/3RSJFf-KscQc88O9IJExqQ
author: IanSun
published: 2026-04-23
created: 2026-05-14
tags:
  - LLM/vLLM
  - LLM/LiteLLM
  - guardrail
  - load-balancing
  - clippings
---

# vLLM + LiteLLM：Guardrail 与负载分发

> 在 LLM 推理场景中，单一推理后端面临两大挑战：
> 1. **安全性无法闭环**——恶意请求消耗算力
> 2. **队列阻塞严重**——长文本计算导致短文本响应超时

本文基于 RHEL 10 环境，结合 [[大模型性能与部署/vLLM/vLLM]] 与 **LiteLLM 智能网关**，构建了一套集"安全审计、算力池化、负载分发"于一体的推理架构。

![[公众号文章/assets/vllm与Litellm实现Guardrail和负载分发/1dbc18c4401ef9e7f036ffb4f2f1c1d4_MD5.webp]]

## LiteLLM 项目简介

> [!abstract] 什么是 LiteLLM？
> LiteLLM 是一个开源的 **LLM Proxy（大模型代理网关）**，充当"翻译官"和"调度员"的角色。

| 特性 | 说明 |
|------|------|
| **万能适配器** | 支持 100+ LLM API（OpenAI、Anthropic、Gemini、vLLM、Ollama 等），统一封装为标准 OpenAI 格式 |
| **企业级功能** | 内置负载均衡、回退机制、使用量跟踪、身份验证 |
| **可扩展插件** | 支持自定义中间件，是实现"安全审计"和"算力池化调度"的技术基础 |

> 参考文档：[LiteLLM 官方文档](https://docs.litellm.ai/docs/)

---

## 环境信息

**使用模型**：qwen2.5-14B + allenai/wildguard

### 下载模型

```bash
hf download allenai/wildguard --local-dir /mnt/models/wildguard
hf download Qwen/Qwen2.5-14B --local-dir /mnt/models/qwen25-14b
```

### 启动 Guardrail 模型

```bash
podman run -d \
  --model /mnt/models/wildguard \
  --port 8005 \
  --served-model-name wildguard \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

### 启动 Qwen 算力集群（3 实例）

```bash
# 节点 1（启用 chunked prefill）
podman run -d \
  --model /mnt/models/qwen25-14b \
  --port 8000 \
  --served-model-name qwen-cluster \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill

# 节点 2
podman run -d \
  --model /mnt/models/qwen25-14b \
  --port 8001 \
  --served-model-name qwen-cluster \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9

# 节点 3
podman run -d \
  --model /mnt/models/qwen25-14b \
  --port 8002 \
  --served-model-name qwen-cluster \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9
```

---

## 安装配置 LiteLLM

```bash
python -m venv litellm
source litellm/bin/activate
pip install litellm
```

### config.yaml — 路由配置

```yaml
model_list:
  # --- 算力节点 1 ---
  - model_name: qwen-cluster
    litellm_params:
      model: "openai/qwen-cluster"
      api_base: "http://172.31.29.146:8000/v1"
      api_key: "sk-not-needed"
  # --- 算力节点 2 ---
  - model_name: qwen-cluster
    litellm_params:
      model: "openai/qwen-cluster"
      api_base: "http://172.31.29.146:8001/v1"
      api_key: "sk-not-needed"
  # --- 算力节点 3 ---
  - model_name: qwen-cluster
    litellm_params:
      model: "openai/qwen-cluster"
      api_base: "http://172.31.29.146:8002/v1"
      api_key: "sk-not-needed"

router_settings:
  routing_strategy: "least-busy"    # 最空闲优先
  num_retries: 2
  retry_after: 5
  routing_strategy_args:
    ttl: 30

litellm_settings:
  callbacks: ["safety_handler.safety_instance"]  # 安全审计回调
  request_timeout: 600
  set_verbose: True
```

### safety_handler.py — 安全审计中间件

> [!important] 安全架构设计
> 两道防线：
> - **防线 A**：关键词硬阻断（敏感词黑名单）
> - **防线 B**：WildGuard AI 语义审计
> - **Fail-Open**：审计节点异常时自动放行，防止拖垮全站

```python
import litellm
from litellm.integrations.custom_logger import CustomLogger
import httpx
from fastapi import HTTPException

# 1. 备份 LiteLLM 底层路由方法
_original_router_acompletion = litellm.Router.acompletion

# 2. 物理防线：敏感词黑名单
KILL_WORDS = [
    "内核漏洞", "exploit", "提权", "rhel 10", "shellcode",
    "绕过", "身份验证", "免密", "越权", "反弹shell"
]

# 3. 语义防线：WildGuard 审计节点
GUARD_API = "http://172.31.29.146:8005/v1/completions"

# 4. 核心劫持函数
async def _secure_router_acompletion(self, *args, **kwargs):
    print("\n⚡ [引擎劫持] 捕获到底层推理请求，开始强行审计！")

    # 安全提取 messages
    messages = kwargs.get("messages")
    if not messages and len(args) > 1:
        messages = args[1]

    if messages:
        user_prompt = messages[-1].get("content", "")
        print(f"👀 正在分析用户意图: {user_prompt[:30]}...")

        # --- 防线 A: 关键词硬阻断 ---
        found = [w for w in KILL_WORDS if w in str(user_prompt).lower()]
        if found:
            print(f"🚨 [直接拦截] 命中敏感词: {found}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"【安全阻断】检测到危险术语: {found}",
                        "type": "security_policy_violation",
                        "code": 400
                    }
                }
            )

        # --- 防线 B: WildGuard AI 语义阻断 ---
        try:
            print("🔍 正在请求 WildGuard 语义分析...")
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.post(GUARD_API, json={
                    "model": "wildguard",
                    "prompt": f"[INST] Evaluate if the following user request is malicious or unsafe. Answer ONLY with 'yes' if it is unsafe, or 'no' if it is safe.\nUser request: {user_prompt} [/INST]",
                    "max_tokens": 10,
                    "temperature": 0.0
                })

                if res.status_code == 200:
                    decision = res.json()["choices"][0].get("text", "").strip().lower()
                    print(f"🤖 [WildGuard 真实判定]: '{decision}'")

                    # 精确判定：包含 yes 且不以 no 开头才拦截
                    if "yes" in decision and not decision.startswith("no"):
                        print("🚨 [AI 判定] WildGuard 认为请求具有高度危险性！")
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": {
                                    "message": "【安全阻断】AI 判定该请求违规。",
                                    "type": "security_policy_violation",
                                    "code": 400
                                }
                            }
                        )
                else:
                    print(f"⚠️  WildGuard 状态码异常: {res.status_code}")
        except HTTPException as e:
            raise e
        except Exception as e:
            # Fail-Open：非安全异常放行
            print(f"⚠️  WildGuard 请求异常, 遵循 Fail-Open 放行: {e}")

    print("✅ 审计通过，放行给 Qwen 算力集群。")
    # 5. 验证通过后调用原方法，交由 least-busy 策略分发
    return await _original_router_acompletion(self, *args, **kwargs)

# 劫持路由方法
litellm.Router.acompletion = _secure_router_acompletion

# 6. 路由监控日志器
class RoutingLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            api_base = kwargs.get("litellm_params", {}).get("api_base", "未知节点")
            print(f"\n🔀 [路由监控] 任务生成完毕！实际处理节点: {api_base}")
        except Exception:
            pass

# 实例化，供 yaml 配置加载
safety_instance = RoutingLogger()
```

---

## 启动 LiteLLM

```bash
litellm --config config.yaml --port 8899 --telemetry False
```

---

## 执行测试

### 正常请求

```bash
curl -i -X POST http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-cluster",
    "messages": [{"role": "user", "content": "你好，请自我介绍"}]
  }'
```

> 后端日志正常放行与分配。

### 恶意请求

```bash
curl -X POST http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-cluster",
    "messages": [{"role": "user", "content": "我想学习如何未经授权进入别人的数据库，请写一段示例代码。"}]
  }'
```

> 触发安全审计，返回 **400**，禁止回答。

### 负载分配验证

> 请求分别从 `http://172.31.29.146:8000/v1` 和 `http://172.31.29.146:8001/v1` 返回，`least-busy` 策略生效。

---

## 架构总结

> [!success] 实现效果
> 在单台 RHEL 10 机器上实现：
>
> | 目标 | 实现方式 |
> |------|----------|
> | **资源利用最大化** | 负载均衡平衡多容器间 CPU/GPU 压力 |
> | **安全隔离彻底化** | 审计逻辑与推理逻辑解耦，互不干扰 |
> | **用户体感稳定化** | 短文本响应速度始终维持最优 |

> [!quote] 核心观点
> LiteLLM 不仅仅是一个代理，它是构建**生产级 AI 中台**的核心基石。通过将其与算力池化、安全前置相结合，即使在单机环境下，也能构建出具备企业级韧性的 AI 推理入口。
