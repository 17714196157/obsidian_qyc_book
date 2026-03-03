---
title: vLLM v0.16.0 深度解读：异步调度 + 流水线并行如何让推理吞吐量提升30%
source: https://mp.weixin.qq.com/s/4ulJZdiwmt0GEsIOIGCuyQ
created: 2026-03-01
tags:
  - 公众号文章
  - vllm
---
![[公众号文章/assets/vLLM v0.16.0 深度解读：异步调度 + 流水线并行如何让推理吞吐量提升30%/f48619a81ca5f97ab6300d35558603d2_MD5.jpg]]
本文基于 vLLM v0.16.0 官方 Release Notes 深度解读  
  
一、为什么说这是"生产级"的重要更新？  
  
2026年2月，vLLM 发布了 v0.16.0 版本。这个版本最让我关注的不是又支持了多少新模型，而是一个底层架构的改进：Async Scheduling + Pipeline Parallelism 的完全整合。  
  
官方数据显示，这个改进带来了：  
• 端到端吞吐量提升 30.8%  
• TPOT（Time Per Output Token）降低 31.8%  
  

对于在生产环境跑大模型推理的同学来说，这意味着什么？同样的硬件，能服务更多用户；同样的并发，用户等待时间更短。  
  
二、先理解问题：传统流水线并行的痛点  
  
在大模型推理中，当模型太大单卡放不下时，我们会用 Pipeline Parallelism（流水线并行） ——把模型按层切分，不同层放到不同GPU上，像流水线一样处理请求。  
  
传统的问题在于：调度是同步的。Scheduler 必须等前一阶段完全执行完，才能调度下一阶段。这就导致了 GPU 的"空闲等待"（Bubble）。

  

三、Async Scheduling：让 Scheduler "异步起来"  
  
v0.16.0 的核心改进是 Async Scheduling（异步调度）。  
  
简单说，就是让 Scheduler 不必等待前一阶段完成，而是可以"预判"和"预调度"。结合 Pipeline Parallelism，实现了真正的流水线化：  
  
传统方式：  
GPU 0: \[请求A层1-10\] -> 等待 -> \[请求B层1-10\] -> 等待 ->...  
GPU 1: 等待 -> \[请求A层11-20\] -> 等待 -> \[请求B层11-20\] ->...  
  
Async + PP：  
GPU 0: \[请求A层1-10\] -> \[请求B层1-10\] -> \[请求C层1-10\] ->...  
GPU 1: 等待 -> \[请求A层11-20\] -> \[请求B层11-20\] ->...  
↑ 这里的等待被压缩到最小  
  
关键点在于： Scheduler 现在可以在一个请求的前一阶段还在执行时，就提前把后续阶段加入调度队列。GPU 间的数据传输和计算重叠度大幅提升。  
  
四、30% 性能提升从哪来？  
  
1\. 减少了 Pipeline Bubble  
  
传统的流水线并行，bubble（空闲时间）占比可能达到 30-50%。Async Scheduling 通过预调度和执行重叠，把这个 bubble 压到了更低。  
  
2\. 更好的内存管理  
  
异步调度需要更复杂的内存管理。vLLM 的 PagedAttention 在这里发挥了优势 —— 细粒度的 KV Cache 管理让异步调度可以灵活地分配和回收显存。  
  
3\. TPOT 降低的本质  
  
TPOT（每个输出token的耗时）降低 31.8%，意味着用户体验的直接改善。对于实时性要求高的应用（比如聊天机器人、代码补全），这个改进很关键。

  

五、如何启用？  
  
好消息是，如果你已经在用 Pipeline Parallelism，升级到 v0.16.0 后这个优化是 默认开启 的。  
  
启动参数保持不变：  
  
vllm serve meta-llama/Llama-2-70b \\  
\--pipeline-parallel-size 4 \\  
\--tensor-parallel-size 2  
  
vLLM 会自动检测并启用 Async Scheduling。  
  
六、其他亮点  
  
• Realtime API ：WebSocket 音频流，基于 Voxtral 架构  
• RLHF 优化 ：NCCL 权重同步、层级别重载、引擎暂停/恢复  
• Speculative Decoding ：统一并行草稿，支持结构化输出  
  

参考vllm-project/vllm/releases/tag/v0.16.0  