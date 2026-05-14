---
title: "大模型推理，加速王者：DFlash，即将支持 DeepSeek-V4"
source: "https://mp.weixin.qq.com/s/4GceWUQdABzBE7BLCcFGOA"
author:
  - "[[老章很忙]]"
published:
created: 2026-05-10
description:
tags:
  - "clippings"
---
原创 老章很忙 *2026年5月10日 16:36*

前文介绍了 [谷歌亮招，Gemma 4 加速 3 倍，vLLM Day0 支持](https://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&mid=2649013393&idx=1&sn=dd6981638bd3d35202dfeb3f2d522680&scene=21#wechat_redirect)

然后发现 DFlash 更猛：加速高达 6 倍 ⚡

同等输出质量、完全无损，开源、即插即用 ⚡

<video src="https://mpvideo.qpic.cn/0bc3taahsaaaj4ag6jiaknvfbggdpgmaa6ia.f10002.mp4?dis_k=0c7f6de5e5a56bede0ae8a50255e6358&amp;dis_t=1778407611&amp;play_scene=10120&amp;auth_info=Xq6AjBoWL0aX8PaQaDUfHF8Vc3Q1BTATCGIMYzFtVE9iLnIfMXxGMj4bKhhNJw==&amp;auth_key=ae105a491c91761bf45ae10ca6a3e6b2&amp;vid=wxv_4509690037848096769&amp;format_id=10002&amp;support_redirect=0&amp;mmversion=false" controls="">您的浏览器不支持 video 标签</video>

DFlash 其实我之前简单介绍过： [\# 一个更神奇的 Qwen3.5-27B 版本，推理速度暴涨 5 倍](https://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&mid=2649012347&idx=1&sn=c3ba12b69e35260eb14d4f47c12ab15d&scene=21#wechat_redirect)

本文事无巨细把 **DFlash** 这个项目掰开揉碎讲一遍

### 简介

大模型生成的本质是「自回归」 —— 第 N 个 token 必须等第 N-1 个 token 算完才能开始，token 之间是 **串行** 的，怎么也快不起来

业界目前最主流的解法叫 **Speculative Decoding（投机解码）** ：

1. 找一个小的 draft 模型先飞快地"猜"出一串 token
2. 大的 target 模型并行验证这串 token
3. 验证通过的直接采纳，验证不过的丢掉重来

理论上能把吞吐拉很高，但目前最强的 EAGLE-3 也只能做到 2-3× 加速， **因为 draft 模型自己也是自回归的** ，仍然是一个 token 一个 token 出，draft 这步本身就是瓶颈

DFlash 干的事很狠： **把 draft 模型从自回归换成了 block diffusion（块扩散）模型**

一次前向传播直接生成一整块 16 个 token，不再串行

结果就是：

- Qwen3-8B 上做到 **6× 无损加速**
- 比 EAGLE-3 快 **2.5×**
- 推理模型（开 thinking）上也有 **4.5×** 加速

<video src="https://mpvideo.qpic.cn/0bc3hefjmaak4uaiauikmzvfuoodsy4qvfqa.f10002.mp4?dis_k=4ebc2b416dd1a4af6dd32445781cee7e&amp;dis_t=1778407611&amp;play_scene=10120&amp;auth_info=WOjIwhwSKROS9PDBbGZOGVtCJXA3VmQWXmhZYjRoWUtkfSQdbHhAZzsfLElJdA==&amp;auth_key=95aaff4df0e2adf35d8ce9968b22d20a&amp;vid=wxv_4509690118747914245&amp;format_id=10002&amp;support_redirect=0&amp;mmversion=false" controls="">您的浏览器不支持 video 标签</video>

### 核心思路

#### 关键洞察：Target 模型的隐藏特征里藏着未来

直接把 diffusion 模型缩小当 drafter，效果其实很一般（5 层的朴素扩散 drafter 加速只有 3× 左右）—— 因为它太小了，想从零预测未来 token 不现实

但作者发现了一个白吃的午餐： **大的 target 模型在生成第 N 个 token 时，hidden states 里其实已经隐含了第 N+1、N+2、N+3...的信息**

那思路就清晰了 —— 把 target 的 hidden features 喂给 draft，让 draft "站在巨人的肩膀上"猜，而不是从零猜

#### 为什么 diffusion 才是最佳形态

自回归 drafter 的成本随 token 数 **线性** 增长，所以 EAGLE-3 不得不把网络砍到只剩 1 层 transformer，质量自然受限

扩散 drafter 一次前向出全部 token，成本和 token 数 **几乎无关**

> ❝
> 
> 一个多层的 DFlash 生成 16 个 token， **比 1 层 EAGLE-3 生成 8 个 token 还快**

更深的网络 + 更多的 token + 更低的延迟，听起来像作弊但确实成立

![draft 模型直接复用 target 的 embedding 和 LM head，只有中间几层是新训练的，参数量保持极低](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

draft 模型直接复用 target 的 embedding 和 LM head，只有中间几层是新训练的，参数量保持极低

整套流程分三步：

1. **Feature Fusion** ：从 target 模型多层均匀采样 hidden features，经过一个轻量级投影融合
2. **KV Injection** ：融合后的特征直接注入 draft 模型 **每一层** 的 K/V 投影里，存进 KV cache —— 这是和 EAGLE-3 最关键的区别。EAGLE-3 只在第一层喂特征，越往后稀释越严重；DFlash 每层都灌，acceptance length 随深度 **正向** 增长
3. **Parallel Drafting** ：基于这套丰富 context 一次性预测下一块 token

### 实测数据

Qwen3-8B 在各类 benchmark 上的 greedy decoding 加速倍数（DFlash 用 block 16 + 1 步去噪，EAGLE-3 用 spec 长度 7）：

| 任务 | 原速 | EAGLE-3 | DFlash |
| --- | --- | --- | --- |
| GSM8K | 1× | 2.13× | **5.20×** |
| MATH-500 | 1× | 2.18× | **6.17×** |
| AIME24 | 1× | 2.25× | **5.91×** |
| AIME25 | 1× | 2.18× | **5.85×** |
| HumanEval | 1× | 2.48× | **5.20×** |
| MBPP | 1× | 2.27× | **4.75×** |
| LiveCodeBench | 1× | 2.24× | **5.43×** |
| SWE-Bench | 1× | 1.90× | 2.92× |
| MT-Bench | 1× | 1.94× | 2.79× |
| Alpaca | 1× | 1.88× | 2.27× |

数学和代码场景下，DFlash 直接是 EAGLE-3 的 2 倍以上速度

温度采样（temp=1）以及开 thinking 模式下，DFlash 同样有 ~4.5× 的稳定加速

### 已支持的模型

DFlash 把现在的开源主力基本兜住了：

| 类别 | 模型 |
| --- | --- |
| Gemma 系列 | gemma-4-26B-A4B-it / gemma-4-31B-it |
| Qwen 系列 | Qwen3.6-27B / Qwen3.6-35B-A3B / Qwen3.5-4B/9B/27B/35B-A3B/122B-A10B |
| Coder 系列 | Qwen3-Coder-Next / Qwen3-Coder-30B-A3B |
| 大厂模型 | MiniMax-M2.5（preview）/ Kimi-K2.5 |
| OSS | gpt-oss-20b / gpt-oss-120b |
| 即将到来 | DeepSeek-V4-Flash / V4-Pro / MiniMax-M2.7 / GLM-5.1 |

要新模型？GitHub 提 issue 就行，作者也表示 **会开源训练 recipe** ，到时候你自己训一个 draft 模型加速任意 LLM

### 安装

DFlash 同时支持四个后端 —— vLLM、SGLang、Transformers、MLX（M 系列 Mac），按需选

```
# Transformers
uv pip install -e ".[transformers]"

# SGLang
uv pip install -e ".[sglang]"

# vLLM（v0.20.1+ 已经核内核支持）
uv pip install -e ".[vllm]"

# MLX（Apple Silicon）
pip install -e ".[mlx]"
```

Gemma 4 的 vLLM 支持还在 PR 阶段，作者直接给了 docker 镜像：

```
docker pull ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130
```

### 使用

#### vLLM 启服务（以 Qwen3.5-27B 为例）

```
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

#### Gemma 4 用 docker 一把梭

```
docker run --rm -it \
  --gpus all --ipc=host --shm-size=16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130 \
  google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 --port 8000 \
  --speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 15, "attention_backend": "flash_attn"}' \
  --attention-backend triton_attn \
  --max-num-batched-tokens 32768 \
  --trust-remote-code
```

#### SGLang

```
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-35B-A3B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 \
  --attention-backend trtllm_mha \
  --speculative-draft-attention-backend fa4 \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

#### Transformers (Qwen3 / LLaMA-3.1)

```
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

draft  = AutoModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16",
            trust_remote_code=True, dtype="auto", device_map="cuda:0").eval()
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",
            dtype="auto", device_map="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False).to(draft.device)

output = draft.spec_generate(
    input_ids=input_ids, max_new_tokens=2048, temperature=0.0,
    target=target, stop_token_ids=[tokenizer.eos_token_id])
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

#### MLX（M5 Pro 实测可用）

```
from dflash.model_mlx import load, load_draft, stream_generate

model, tokenizer = load("Qwen/Qwen3.5-4B")
draft = load_draft("z-lab/Qwen3.5-4B-DFlash")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)

tps = 0.0
for r in stream_generate(model, draft, tokenizer, prompt,
                         block_size=16, max_tokens=2048, temperature=0.6):
    print(r.text, end="", flush=True)
    tps = r.generation_tps
print(f"\nThroughput: {tps:.2f} tok/s")
```

Mac 用户终于能在本地享受到 5× 推理加速了，这条对苹果用户非常友好

### 总结

DFlash 这个项目最大的价值，老章总结成一句话：

> ❝
> 
> **它把扩散模型的角色重新定义了** —— 扩散模型不需要去和自回归 LLM 比生成质量，它只要做好「极快极准的 drafter」就够了，质量由 target 模型最后做投机验证来兜底

适合谁用？

- **生产环境部署 LLM 的同学** ：vLLM / SGLang 都已经原生支持，加个 `--speculative-config` 就能上，开发成本极低
- **手头有 Apple Silicon 的同学** ：MLX 后端实测可用，本地大模型一夜之间快 5 倍
- **做推理加速 / 投机解码方向研究的同学** ：论文 + 代码 + 训练 recipe 即将全开源，是个好的二次创新基础

不适合谁？

- 单卡显存吃紧的同学要注意，draft 模型也要占显存
- 极小模型（< 3B）加速空间本身就不大，性价比一般

支持的模型清单还在快速扩张，DeepSeek-V4 系列、GLM-5.1 都在 coming soon 里，未来一段时间值得持续关注

#DFlash #推理加速 #投机解码 #BlockDiffusion #vLLM

**制作不易，如果这篇文章觉得对你有用，可否点个关注。给我个三连击：点赞、转发和在看。若可以再给我加个🌟，谢谢你看我的文章，我们下篇再见！**
