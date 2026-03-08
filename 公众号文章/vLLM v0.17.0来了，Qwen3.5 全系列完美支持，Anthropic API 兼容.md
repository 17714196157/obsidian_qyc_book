---
title: "vLLM v0.17.0来了，Qwen3.5 全系列完美支持，Anthropic API 兼容"
source: "https://mp.weixin.qq.com/s/z6f2GFxppfq24qnGvvQDMg"
author:
  - "[[老章很忙]]"
published:
tags:
  - "clippings"
---
原创 老章很忙 *2026年3月7日 17:51*

  

关于 vLLM，我之前写过不少：

- [vLLM 重磅项目](http://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&mid=2649007097&idx=1&sn=976c788bd136675b0afd2196068acd51&chksm=879331d3b0e4b8c5b22bb95ed6518c2dcb076903a4be17e73d58f0f627283ee6155f3ab3ee73&scene=21#wechat_redirect)
- [大模型本地部署，vLLM 睡眠模式来了](http://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&mid=2649004244&idx=1&sn=873272ec310f8da7376c6574fdd48282&chksm=87930afeb0e483e8dddce30ed309ee8d95dc41971b4c1f3a3d0aa0741e5807d7e33c619e1c28&scene=21#wechat_redirect)
- [vLLM 最新版来了，Docker Model Runner 集成vLLM](http://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&mid=2649005048&idx=1&sn=e0e20c72e83cf64c8c4c481fdedd9411&chksm=879309d2b0e480c4f23a4b6671812e6f1587fc6fe10d419cb9c2480a86b09122fa79ba9df253&scene=21#wechat_redirect)
- [全模态大模型部署，vLLM-Omni 来了，100%开源](http://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&mid=2649006106&idx=1&sn=bc85b16431dbfc5361e14de42cff02e5&chksm=87933230b0e4bb265a5999ad557f1c8072ef65540075eddc18c7cf03b5c194cfeb2e20652167&scene=21#wechat_redirect)

今天 vLLM **v0.17.0 正式发布**

![[公众号文章/assets/vLLM v0.17.0来了，Qwen3.5 全系列完美支持，Anthropic API 兼容/821ee13522536ba26d8059eadc27e6b7_MD5.webp]]

### 十大核心亮点速览

我从 Release Notes 里提炼了 v0.17.0 最值得关注的 **十大核心亮点** ，按重要程度排列：

---

#### 1️⃣ FlashAttention 4 集成

这可能是这个版本最让人兴奋的更新。vLLM 现在正式支持 **FlashAttention 4 后端** 了。

FlashAttention 一路从 1 到 2 到 3，现在 4 也来了。每一代都在推动 attention 计算的效率极限。FA4 在前代基础上又做了大量底层优化，对于长序列、大模型的推理性能提升显著。

如果你在用 H100/H200 或者更新的 GPU 跑大模型推理，升级到 v0.17 应该能明显感受到速度提升。

#### 2️⃣ Model Runner V2 里程碑：全面成熟

Model Runner V2 是 vLLM 下一代模型执行架构，在这个版本中达到了一个 **重要的成熟里程碑** ：

- **Pipeline Parallel** （流水线并行）
- **Decode Context Parallel** （解码上下文并行）
- **Eagle3 推测解码** \+ CUDA Graph
- **Pooling 模型支持**
- **分段 & 混合 CUDA Graph 捕获**
- **DP+EP 推测解码**
- **全新 ModelState 架构**

此外官方还发布了 **Model Runner V2 的设计文档** ，对于想深入了解 vLLM 内部架构的同学，这是一份非常好的学习资料。

简单来说，Model Runner V2 是 vLLM 的「心脏升级」。它让 vLLM 在多卡、多节点、各种并行策略下的推理变得更加灵活和高效。

#### 3️⃣ Qwen3.5 全家桶支持

我之前介绍过的方法，vLLM一节都是用的nightly版（ [Qwen3.5 0.8B/2B/4B/9B 小模型本地部署指南，微调教程](https://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&mid=2649010151&idx=1&sn=7b330690a60d5758fa6167e6c158748b&scene=21#wechat_redirect) ）

**Qwen3.5 模型全系列** 在这个版本得到了完整支持，包括：

- 基于 **GDN（Gated Delta Networks）** 的全新架构
- **FP8 量化** 支持
- **MTP 推测解码**
- **推理解析器** （reasoning parser）支持

这意味着你可以直接在 vLLM 上跑 Qwen3.5 的各种版本,享受推测解码和量化加速的全套优化。

对于国内用户来说，这可能是最实际的更新之一——Qwen3.5 是目前开源圈里最强的中文大模型之一，但是这一波 vLLM 有点慢了。

#### 4️⃣ --performance-mode 一键性能调优

这个功能太贴心了。之前部署 vLLM，性能调优需要手动设置一堆参数（batch size、调度策略等），对新手非常不友好。

现在只需要一个参数：

```
vllm serve your-model --performance-mode throughput
```

提供三种模式：

- \*\* `balanced` \*\*：均衡模式，适合大多数场景
- \*\* `interactivity` \*\*：交互模式，优先降低首 token 延迟，适合聊天场景
- \*\* `throughput` \*\*：吞吐模式，最大化吞吐量，适合批处理场景

不用再去查文档翻参数了，一个 flag 搞定。这种 **把复杂度封装起来** 的思路，我很喜欢。

#### 5️⃣ Anthropic API 兼容

vLLM 之前一直兼容 OpenAI API 格式，现在开始支持 **Anthropic API 兼容** 了：

- `thinking blocks` （思考块）支持
- `count_tokens` API
- `tool_choice=none` 选项
- streaming 和图片处理修复

这意味着如果你的应用代码之前是基于 Anthropic Claude API 写的，现在可以 **无缝切换到本地 vLLM 部署的模型** 。API 兼容性做得越来越好，这对于降低迁移成本太重要了。

#### 6️⃣ 权重卸载 V2：预取技术隐藏延迟

对于显存不够用的同学，这个更新很关键。

v0.17 的权重卸载器引入了 **预取机制** （Prefetching），可以在 GPU 计算的同时，把下一层的权重从 CPU 加载到 GPU，从而 **隐藏权重加载延迟** 。

此外还支持了：

- **选择性 CPU 权重卸载** ：不用全部卸载，只卸载你指定的层
- **无需双倍 pinned memory 的 CPU 卸载** ：省内存

这对于在消费级 GPU（3090、4090）上跑大模型的同学来说，是实打实的优化。

#### 7️⃣ 弹性专家并行 Phase 2

**MoE（Mixture of Experts）模型** 是当前大模型的主流架构（DeepSeek-V3/V3.2、Qwen3 MoE、Llama 4 等），vLLM 在这个版本引入了 **弹性专家并行 Milestone 2** 。

核心能力： **动态 GPU 缩放** 。

什么意思？就是你的 MoE 模型可以根据负载动态调整使用的 GPU 数量，负载低的时候少用几张卡省钱，负载高的时候自动扩展。这对于生产环境的成本优化太重要了。

#### 8️⃣ 量化 LoRA 适配器直接加载

之前在 vLLM 上用 LoRA 微调后的模型，如果是量化版本（比如 QLoRA），需要各种周折才能加载。

现在，vLLM 可以 **直接加载量化 LoRA 适配器** 了。

这对于做 LoRA 微调 + 量化部署的工作流来说是个大利好。QLoRA 训练完直接扔到 vLLM 里就能跑，中间环节省了。

#### 9️⃣ 推测解码全面进化

推测解码（Speculative Decoding）是加速 LLM 推理的关键技术，v0.17 在这方面做了大量优化：

- **Eagle3** 推测解码支持 CUDA Graph，速度更快
- **Nemotron-H** MTP 和 Mamba 推测解码
- **Sparse MLA + MTP** 全 CUDA Graph 支持
- **DP+EP** 推测解码（数据并行 + 专家并行）
- Eagle3 支持 **disaggregated serving** （分离式推理）

特别是 Eagle3 + CUDA Graph 这个组合，是这次推测解码部分最值得关注的组合之一。

#### 🔟 Kernel 层面的深度优化

这个版本在底层内核上做了大量「不起眼但很重要」的优化：

- **FlashInfer Sparse MLA** 后端
- **Triton top-k / top-p 采样器内核**
- **TRTLLM DSV3 Router GEMM 内核** ：batch-1 场景加速 6%
- **FA3 swizzle 优化**
- **256-bit LDG/STG 激活内核**
- **Helion 内核框架** ：自动调优基础设施

这些优化可能单个看不起眼，但加在一起就是量变引起质变。实际测试中，DeepSeek R1 BF16 最低延迟 QKV GEMM 做到了 **0.5% 端到端加速** ，Pipeline Parallel 异步收发做到了 **2.9% 端到端吞吐提升** ，pooling maxsim 做到了 **13.9% 吞吐提升** 。

### 硬件支持：不止 NVIDIA

vLLM 越来越不是 NVIDIA 的专属了。v0.17 在硬件支持上做了大量工作：

**NVIDIA 方面：**

- SM100（Blackwell）FP8 MLA prefill 支持
- SM100 MXFP8 块级缩放分组矩阵乘法
- SM120 FP8 GEMM 优化
- FlashInfer DeepGEMM 在 SM90 上默认开启 swapAB

**AMD ROCm 方面：**

- AITER 融合 RoPE+KVCache
- gfx950 上 MXFP4 MoE 权重预混洗
- bitsandbytes 量化支持
- CK（Composable Kernel）MoE 量化后端

**Intel XPU 方面：**

- CUDA graph 支持终于来了
- NIXL GPUDirect RDMA

**CPU 方面：**

- ARM BF16 交叉编译
- s390x FP16 支持
- 同时支持 AVX2 和 AVX512 的 CPU 发行版

如果你是 AMD 或 Intel 的用户，现在上 vLLM 的体验已经好了很多。虽然和 NVIDIA 比还有差距，但差距在快速缩小。

### ASR 模型支持：不只是 LLM 了

v0.17 有一个很有意思的变化——开始支持 **ASR（语音识别）模型** 了：

- **FunASR**
- **FireRedASR2**
- **Qwen3-ASR 实时流式识别**

vLLM 从名字看是「vLLM」——Virtual LLM，但现在它的野心显然不止于文本大模型。之前加了多模态（视觉、音频），现在又加了 ASR，正在进化成一个 **全模态推理引擎** 。

### 升级注意事项

在你兴冲冲跑去升级之前，说几个需要注意的点：

**1\. PyTorch 2.10 升级（Breaking Change！）**

v0.17 升级到了 PyTorch 2.10，这是 **环境依赖的破坏性变更** 。如果你的环境依赖特定版本的 PyTorch，需要做好兼容性测试。

**2\. CUDA 12.9+ 已知问题**

如果你在 CUDA 12.9+ 上遇到 `CUBLAS_STATUS_INVALID_VALUE` 错误，可以试试：

```
# 方法 1：清理 LD_LIBRARY_PATH
unset LD_LIBRARY_PATH

# 方法 2：uv 安装
uv pip install vllm --torch-backend=auto

# 方法 3：指定 CUDA 版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

**3\. KV 缓存加载策略变更**

KV load failure policy 默认值从 `recompute` 变为 `fail` 。如果你的部署依赖自动重算行为，需要手动设置回去。

### 安装

![[公众号文章/assets/vLLM v0.17.0来了，Qwen3.5 全系列完美支持，Anthropic API 兼容/84d795f134f2c05668b3ccb095313289_MD5.webp]]

安装很简单，一行命令：

```
uv pip install vllm
```

Docker 用户：

```
docker pull vllm/vllm-openai:v0.17.0
docker run --gpus all \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --env "HF_TOKEN=$HF_TOKEN" \
      -p 8000:8000 \
      --ipc=host \
      vllm/vllm-openai:v0.17.0 \
      --model Qwen/Qwen3-0.6B
```

### 和 SGLang 怎么选？

这是评论区最常被问到的问题之一。我简单说下我的看法：

- **vLLM** ：更成熟，社区更大（GitHub 50k+ stars），硬件兼容性更好，企业级特性更丰富（pipeline parallel、disaggregated serving 等）。适合 **生产环境** 部署。
- **SGLang** ：在某些场景下性能更极致（特别是 DeepSeek 系列模型），API 更现代化。适合 **追求极致性能** 的场景。

两者都是顶级的推理引擎，现在更像是 **Chrome vs Firefox** 的关系——竞争推动了整个行业的进步。

### 总结

vLLM v0.17.0 是一个 **里程碑式的版本** 。FlashAttention 4 集成、Model Runner V2 成熟、Qwen3.5 全面支持、一键性能调优、Anthropic API 兼容……几乎每一个更新都是硬核的工程突破。
