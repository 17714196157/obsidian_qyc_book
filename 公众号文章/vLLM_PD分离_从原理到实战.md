---
title: "vLLM_PD分离_从原理到实战"
source: "https://mp.weixin.qq.com/s/ZabpmiD8P56Z1ZT20g2qpw"
author:
  - "[[有点文艺细菌的码]]"
published:
created: 2026-04-14
description: "当你的大模型推理服务遇到首字延迟高、尾延迟抖动大、资源利用率上不去的时候，          PD 分离到底能"
tags:
  - "clippings"
---
原创 有点文艺细菌的码 *2026年4月11日 21:35*

当你的大模型推理服务遇到首字延迟高、尾延迟抖动大、资源利用率上不去的时候，  
PD 分离到底能不能救你，以及怎么救。

## 一、先搞清楚：PD 分离在解决什么问题

大模型推理有两个阶段：

·Prefill（预填充）：把用户输入的全部 token 过一遍模型，生成 KV Cache。计算密集，几乎不占显存带宽，吞吐高但延迟随 prompt 长度线性增长。

·Decode（解码）：逐个生成输出 token，每一步都要读取整个 KV Cache 做注意力计算。访存密集，吞吐受显存带宽限制。

问题出在：这两个阶段的资源需求完全相反，但它们被塞在同一个 GPU 上、同一个调度循环里。

## 合体部署的三宗罪

### 1\. 尾延迟不可控

这是最痛的点。Decode 阶段本来是稳定的逐 token 输出，但调度器随时可能往 batch 里塞一个长 prompt 做 Prefill。这个 Prefill 请求会吃掉大量计算资源，导致正在 Decode 的请求停摆——用户看到的就是输出突然卡顿。

Chunked Prefill 能缓解这个问题，但实际操作中 chunk size 很难调对。vLLM 官方文档原话：

"Chunked prefill with a proper chunk size also can achieve the same goal, but in practice it's hard to figure out the correct chunk size value. So disaggregated prefilling is a much more reliable way to control tail ITL."

### 2\. TTFT 和 ITL 无法独立优化

Prefill 阶段想要更多计算力（加 TP、加 PP）来降低首字延迟；Decode 阶段想要更多显存带宽（可能反而需要小 TP）来提高吞吐。但在合体部署中，你选的并行策略必须同时服务两个阶段，结果就是两头不讨好。

### 3\. GPU 资源浪费

Prefill 吃计算力、Decode 吃带宽，同一个 GPU 很难同时打满两者。

![[公众号文章/assets/vLLM_PD分离_从原理到实战/9d634c3d019897ad97b53a0b055179af_MD5.webp]]

图1：合体部署 vs PD分离 对比

## PD 分离的核心思路

把 Prefill 和 Decode 拆到不同的 vLLM 实例，各自跑在不同的 GPU 上：

·Prefill 实例：专做预填充，只产出 KV Cache，不负责解码。可以配大 TP/PP，全力压计算吞吐。

·Decode 实例：专做解码，从 Prefill 实例接收 KV Cache 后逐 token 输出。可以配小 TP，专注带宽效率。

·KV Cache 传输：通过 KV Connector 将 Prefill 实例计算好的 KV Cache 传输给 Decode 实例。

结果：

·TTFT 和 ITL 可以独立调优，互不干扰

·Decode 实例不再被 Prefill 打断，尾延迟可控

·两类实例可以按需扩缩容，资源利用率更高

\[!\] vLLM 官方文档明确指出——PD 分离不会提升总吞吐量（Disaggregated prefill DOES NOT improve throughput）。它的价值在于降低延迟和提升延迟稳定性，不是让你跑更多请求。

## 二、源码剖析：vLLM 是怎么实现 PD 分离的

所有 PD 分离的实现都在 vllm/distributed/kv\_transfer 目录下。核心架构分三层：

## 2.1 配置层：KVTransferConfig

\# vllm/config/kv\_transfer.py  
@config  
class KVTransferConfig:  
kv\_connector: str | None = None # 连接器类型  
kv\_role: KVRole | None = None # kv\_producer / kv\_consumer / kv\_both  
kv\_rank: int | None = None # 0=Prefill, 1=Decode  
kv\_parallel\_size: int = 1 # 并行实例数  
kv\_ip: str = "127.0.0.1" # 连接 IP  
kv\_port: int = 14579 # 连接端口  
kv\_buffer\_device: str =... # 缓冲设备：cuda / cpu / xpu  
kv\_buffer\_size: float = 1e9 # 缓冲区大小（字节）  
kv\_connector\_extra\_config: dict # 连接器额外配置  
kv\_load\_failure\_policy: str = "fail" # KV 加载失败策略：recompute / fail

关键字段解读：

·kv\_role：决定这个 vLLM 实例的角色：kv\_producer 是 Prefill，kv\_consumer 是 Decode，kv\_both 同时兼做两者

·kv\_rank：在 P2pNcclConnector 中必须指定，0 给 Prefill、1 给 Decode

·kv\_load\_failure\_policy：实用的容错开关——设成 recompute 时，如果 KV Cache 加载失败，Decode 实例会重新做 Prefill 而不是直接报错

## 2.2 调度层：Scheduler 中的 Connector

\# vllm/v1/core/sched/scheduler.py（简化）  
class Scheduler:  
def \_\_init\_\_(self,...):  
if self.vllm\_config.kv\_transfer\_config is not None:  
self.connector = KVConnectorFactory.create\_connector(  
self.vllm\_config, role=KVConnectorRole.SCHEDULER)  
  
def \_schedule(self,...):  
if self.connector:  
num\_new\_matched\_tokens = (  
self.connector.get\_num\_new\_matched\_tokens(request,...))  
\# 如果远程已有部分 KV Cache，可以跳过这些 token 的计算

调度器的工作流程：

·请求进来时，先问 Connector：远程有没有这个请求的 KV Cache？

·如果有，只需要计算新增部分的 token，已有的 KV Cache 可以直接复用

·调度完成后，把 KV Cache 传输的元数据打包到 SchedulerOutput 里

·Worker 侧的 Connector 根据元数据执行实际的 KV Cache 收发

## 2.3 执行层：Worker Connector 与注意力模块

Prefill Worker:  
Layer 0: 计算 Attention -> 存储 KV Cache -> 传输  
Layer 1: 计算 Attention -> 存储 KV Cache -> 传输  
...  
  
Decode Worker:  
Layer 0: 接收 KV Cache -> 加载到显存 -> 计算 Attention  
Layer 1: 接收 KV Cache -> 加载到显存 -> 计算 Attention  
...

这种逐层传输的设计意味着 Decode 实例不需要等整个 Prefill 完成——KV Cache 可以边产生边传输，进一步降低端到端延迟。
[[公众号文章/assets/vLLM_PD分离_从原理到实战/431f83711d37f4abb519f099f9e5db40_MD5.png|Open: file-20260414170339799.png]]
![[公众号文章/assets/vLLM_PD分离_从原理到实战/431f83711d37f4abb519f099f9e5db40_MD5.png]]
图2：vLLM PD分离架构与数据流

## 2.4 三层抽象

vLLM 为 PD 分离定义了三层抽象，对应不同的实现路径：

| 抽象层 | API | 适用场景 |
| --- | --- | --- |
| Connector | 完全自定义 | 需要最大灵活性，自己管理 KV Cache 传输、模型输入编辑等 |
| LookupBuffer | insert + drop\_select | 数据库式——插入 KV Cache，按条件取出并删除 |
| Pipe | send\_tensor + recv\_tensor | 点对点管道——类似 torch.distributed 的 send/recv |

\[!\] LookupBuffer 的 drop\_select 是阻塞操作，insert 是非阻塞的。这意味着 Decode 端会阻塞等待 KV Cache 就绪。

## 三、哪些场景该用 PD 分离

## 适合用的

### 1\. 对尾延迟敏感的在线服务

这是 PD 分离最核心的适用场景。如果你的用户在聊天过程中经常遇到输出卡顿（尤其是多个长 prompt 并发时），PD 分离几乎是唯一的可靠解。

### 2\. 长上下文场景

RAG、长文档问答、代码补全——prompt 动辄几千甚至上万 token。Prefill 阶段耗时很长，极易打断 Decode。

### 3\. TTFT 和 ITL 有差异化需求

比如你的业务要求 TTFT < 500ms（Prefill 需要大 TP 压延迟），同时 Decode 需要稳定低 ITL（小 TP 更高效）。合体部署下两者冲突，PD 分离下各管各的。

### 4\. 多模态推理（编码器分离）

vLLM 还支持编码器分离（Disaggregated Encoder），把视觉编码器和语言模型拆到不同实例。编码器轻量级可以独立扩缩容，语言模型可以跳过编码直接处理纯文本请求，编码器输出还能跨实例缓存复用。

## 不适合用的

### 1\. 追求总吞吐量的离线批处理

PD 分离不提升吞吐。如果你做的是离线推理，只关心单位时间处理多少请求，不关心延迟，PD 分离反而增加了 KV Cache 传输的开销。

### 2\. 短 prompt + 短输出的简单场景

prompt 只有几十个 token 时，Prefill 本身就很快，分离的收益微乎其微，还多了传输开销。

### 3\. GPU 资源紧张

PD 分离至少需要 2 张 GPU（1 Prefill + 1 Decode）。如果你只有 1 张卡，或者卡已经全被其他服务占满，PD 分离不是你的菜。

### 4\. 追求稳定生产级部署

vLLM 官方把 PD 分离标记为 experimental（实验性功能），API 随时可能变化。如果你需要稳定的长期部署，建议等它正式 GA。

## 四、怎么用：三种部署模式
[[公众号文章/assets/vLLM_PD分离_从原理到实战/dd7d25527bc86b1d4c010726ecbdc40c_MD5.png|Open: file-20260414170356698.png]]
![[公众号文章/assets/vLLM_PD分离_从原理到实战/dd7d25527bc86b1d4c010726ecbdc40c_MD5.png]]
图4：三种部署模式概览

## 模式一：离线推理（最简单）

适用场景：调试、验证、一次性推理。

vLLM V1 提供了基于 ExampleConnector 的离线 PD 分离，使用本地文件系统做 KV Cache 的中转：

Prefill 端：

from vllm import LLM, SamplingParams  
from vllm.config import KVTransferConfig  
  
llm = LLM(  
model="meta-llama/Llama-3.2-1B-Instruct",  
enforce\_eager=True,  
gpu\_memory\_utilization=0.8,  
kv\_transfer\_config=KVTransferConfig(  
kv\_connector="ExampleConnector",  
kv\_role="kv\_both",  
kv\_connector\_extra\_config={  
"shared\_storage\_path": "local\_storage"  
},  
),  
)  
outputs = llm.generate(prompts, SamplingParams(max\_tokens=1))

Decode 端（另一个进程，可以跑在另一张 GPU 上）：

llm = LLM(  
model="meta-llama/Llama-3.2-1B-Instruct",  
enforce\_eager=True,  
gpu\_memory\_utilization=0.8,  
kv\_transfer\_config=KVTransferConfig(  
kv\_connector="ExampleConnector",  
kv\_role="kv\_both",  
kv\_connector\_extra\_config={  
"shared\_storage\_path": "local\_storage"  
},  
),  
)  
outputs = llm.generate(prompts, SamplingParams(max\_tokens=10))

\[TIP\] ExampleConnector 用本地文件系统做中转，不需要网络通信，适合单机调试。但性能最差，不适合生产。

## 模式二：在线服务 + 代理（生产可用）

适用场景：在线推理服务，需要对外暴露 API。
[[公众号文章/assets/vLLM_PD分离_从原理到实战/5814d4cbaabfa93e5771204db432e1e2_MD5.png|Open: file-20260414170411967.png]]
![[公众号文章/assets/vLLM_PD分离_从原理到实战/5814d4cbaabfa93e5771204db432e1e2_MD5.png]]
图3：在线服务代理转发流程

代理的工作流程：

·收到请求后，先发到 Prefill 实例，把 max\_tokens 改成 1（只做 Prefill）

·Prefill 完成后，KV Cache 通过 Connector 传到 Decode 实例

·代理再把原始请求发到 Decode 实例，流式返回结果

启动 Prefill 实例：

CUDA\_VISIBLE\_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \\  
\--host 0.0.0.0 --port 8100 \\  
\--max-model-len 10000 --gpu-memory-utilization 0.9 \\  
\--kv-transfer-config \\  
'{"kv\_connector":"P2pNcclConnector","kv\_role":"kv\_producer",  
"kv\_rank":0,"kv\_parallel\_size":2,"kv\_buffer\_size":"1e9",  
"kv\_port":"14579"}'

启动 Decode 实例：

CUDA\_VISIBLE\_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \\  
\--host 0.0.0.0 --port 8200 \\  
\--max-model-len 10000 --gpu-memory-utilization 0.7 \\  
\--kv-transfer-config \\  
'{"kv\_connector":"P2pNcclConnector","kv\_role":"kv\_consumer",  
"kv\_rank":1,"kv\_parallel\_size":2,"kv\_buffer\_size":"8e9",  
"kv\_port":"14580"}'

启动代理：

python3 examples/online\_serving/disaggregated\_serving/disagg\_proxy\_demo.py \\  
\--model meta-llama/Llama-3.1-8B-Instruct \\  
\--prefill localhost:8100 --decode localhost:8200 --port 8000

\[!\] 代理当前使用 Round-Robin 调度策略。源码中可以看到 itertools.cycle 做轮询，没有考虑实例负载。生产环境需要更智能的调度。

## 模式三：XpYd 多实例（规模化）

适用场景：多 Prefill + 多 Decode 实例，需要弹性伸缩。

XpYd 表示 X 个 Prefill 实例 + Y 个 Decode 实例。vLLM 提供了 1P3D 的示例脚本：

\# 1P3D：1 个 Prefill + 3 个 Decode  
PREFILL\_GPUS=0 DECODE\_GPUS=1,2,3 \\  
bash examples/online\_serving/disaggregated\_serving\_p2p\_nccl\_xpyd/disagg\_example\_p2p\_nccl\_xpyd.sh

关键配置差异：

·Prefill 实例的 gpu\_memory\_utilization 设为 0.9（吃计算，不囤 KV Cache）

·Decode 实例的 gpu\_memory\_utilization 设为 0.7（留空间存 KV Cache）

·kv\_buffer\_size：Prefill 设为 1e1（几乎不需要缓冲），Decode 设为 8e9（需要大缓冲接收 KV Cache）

\[TIP\] XpYd 的代理支持动态增减实例——通过 /instances/add API 可以在线添加 Prefill 或 Decode 节点。代理会验证新实例的模型是否匹配，然后加入轮询列表。

## 五、连接器选型指南

vLLM 目前注册了 13 种 KV Connector，但常用的就几种：

| 连接器 | 传输方式 | 适用场景 | 成熟度 |
| --- | --- | --- | --- |
| ExampleConnector | 本地文件系统 | 单机调试、离线推理 | 示例级 |
| P2pNcclConnector | NCCL 点对点 | 单机多卡、同机房低延迟 | 生产可用 |
| NixlConnector | NVIDIA NIXL 库 | 异步传输、高性能场景 | 较新 |
| LMCacheConnectorV1 | LMCache + NIXL | 分布式 KV Cache 共享 | 较新 |
| MooncakeConnector | Mooncake 传输 | 生产级分布式 | 社区贡献 |
| FlexKVConnectorV1 | FlexKV 分布式存储 | 超大规模 KV Cache 管理 | 较新 |
| OffloadingConnector | CPU 内存卸载 | 显存不足时用 CPU 缓存 | 辅助 |
| SimpleCPUOffloadConnector | 简单 CPU 卸载 | 最简 CPU offload | 辅助 |
| MultiConnector | 组合多个连接器 | 混合传输策略 | 实验性 |
[[公众号文章/assets/vLLM_PD分离_从原理到实战/bb5eee09e7064d85b3de410463e421e2_MD5.png|Open: file-20260414170426163.png]]
![[公众号文章/assets/vLLM_PD分离_从原理到实战/bb5eee09e7064d85b3de410463e421e2_MD5.png]]
图5：连接器选型对比雷达图

选型建议：

·先跑通：ExampleConnector，零依赖，本地文件中转

·单机多卡：P2pNcclConnector，NCCL 直传延迟最低

·分布式集群：NixlConnector 或 MooncakeConnector，支持跨节点传输

·KV Cache 复用：LMCacheConnectorV1，可以把 KV Cache 缓存在分布式存储中，多个 Decode 实例共享

MultiConnector 的用法比较特殊——它可以组合多个连接器，形成级联传输策略：

\--kv-transfer-config '{  
"kv\_connector":"MultiConnector",  
"kv\_role":"kv\_both",  
"kv\_connector\_extra\_config":{  
"connectors":\[  
{"kv\_connector":"NixlConnector","kv\_role":"kv\_both"},  
{"kv\_connector":"ExampleConnector","kv\_role":"kv\_both",  
"kv\_connector\_extra\_config":{"shared\_storage\_path":"local\_storage"}}  
\]  
}  
}'

## 六、编码器分离：多模态场景的特殊形态

如果你的模型是视觉语言模型（VLM），vLLM 还支持编码器分离——把视觉编码器拆到独立实例：
[[公众号文章/assets/vLLM_PD分离_从原理到实战/4d275d62cc52ecffafd3e08bbcdb4236_MD5.png|Open: file-20260414170439580.png]]
![[公众号文章/assets/vLLM_PD分离_从原理到实战/4d275d62cc52ecffafd3e08bbcdb4236_MD5.png]]
图6：编码器分离架构

配置使用 ECTransferConfig（和 KVTransferConfig 结构对称）：

from vllm.config import ECTransferConfig  
  
ec\_config = ECTransferConfig(  
ec\_connector="ExampleConnector",  
ec\_role="ec\_producer", # 编码器端  
ec\_rank=0,  
ec\_parallel\_size=2,  
)

编码器分离的三个独特价值：

·独立扩缩容：视觉编码器通常很轻量，语言模型很重。分开后可以按需扩缩编码器。

·降低 TTFT：纯文本请求完全跳过编码器，不用等编码器空闲。

·编码输出复用：同一张图片的编码结果可以缓存在共享存储中，任何 Worker 都能取用。

## 七、踩坑实录

## 坑1：kv\_role 忘了填

报错：ValueError: Please specify kv\_role when kv\_connector is set

\# 源码里的校验逻辑：  
if self.kv\_connector is not None and self.kv\_role is None:  
raise ValueError(...)

解法：只要设置了 kv\_connector，就必须同时设置 kv\_role。

## 坑2：P2pNcclConnector 的 kv\_parallel\_size 必须等于 2

P2pNcclConnector 设计上就是 1 对 1 的点对点连接（1P1D），kv\_parallel\_size 只能是 2。如果你设成 3 或 1，NCCL 初始化会失败。

解法：P2pNcclConnector 场景下，kv\_parallel\_size=2 是硬性要求。如果你需要 1P3D，应该用 XpYd 架构——启动多个独立的 P2P 连接对，通过代理做调度。

## 坑3：Decode 实例的 kv\_buffer\_size 设太小

Prefill 实例的 KV Cache 可能很大（长 prompt x 多层 x 多头），如果 Decode 实例的 kv\_buffer\_size 容不下，传输会失败。

解法：Decode 端的 kv\_buffer\_size 要远大于 Prefill 端。从示例脚本看，Prefill 端设 1e1，Decode 端设 8e9——差距三个数量级。

## 坑4：代理用 Round-Robin 不够智能

当前代理的调度策略是简单的轮询（itertools.cycle），不考虑实例负载。如果一个 Decode 实例已经满载，代理还会继续往它发请求。

解法：生产环境需要自己实现调度策略——继承 SchedulingPolicy 类，根据实例的队列深度、延迟等指标做负载感知调度。vLLM 的 PR #15343 正在开发 PDController，未来会提供更智能的调度。

## 坑5：Prefill 实例提前退出导致 Decode 端传输不完整

P2pNcclConnector 模式下，如果 Prefill 进程在 Decode 完成前退出，NCCL 连接会断开，Decode 端收不到完整 KV Cache。

解法：Prefill 端完成 generate 后不要立即退出。离线示例中用了 Event 同步机制——Prefill 等待 prefill\_done 事件后才退出，确保 Decode 端有足够时间接收数据。

## 坑6：HMA 和 Connector 不兼容

如果你启用了混合 KV Cache 管理（HMA），但用的 Connector 不支持 HMA，会直接报错：  
ValueError: Connector XXX does not support HMA but HMA is enabled.

解法：要么换支持 HMA 的连接器，要么加 --disable-hybrid-kv-cache-manager。

## 八、实战检查清单

在决定是否上 PD 分离之前，过一遍这个清单：

\[ \] 你的服务是否对尾延迟敏感？ (是 -> 继续考虑)

\[ \] 你的平均 prompt 长度是否 > 500 tokens？ (是 -> PD 分离收益更大)

\[ \] 你是否需要独立调优 TTFT 和 ITL？ (是 -> PD 分离是唯一解)

\[ \] 你是否有 >= 2 张 GPU 可用？ (是 -> 满足最低硬件要求)

\[ \] 你是否能接受实验性功能？ (PD 分离仍标记为 experimental)

\[ \] 你的 GPU 是否在同一节点或同机房低延迟网络？ (跨机房传输延迟可能抵消分离收益)

全部打勾 -> 可以上了。有否 -> 想清楚再上。

## 九、总结

PD 分离不是银弹，它解决的是一个非常具体但很痛的问题——Prefill 和 Decode 资源需求冲突导致的延迟抖动。

核心结论：

·PD 分离不提升总吞吐量，但可以显著降低 TTFT 和尾延迟

·它适合对延迟敏感、长上下文、需要差异化优化的在线服务

·不适合离线批处理、短 prompt、GPU 资源紧张的场景

·目前仍是 experimental，生产部署需谨慎

·连接器选型从 ExampleConnector 调试，到 P2pNcclConnector 单机，再到 NixlConnector/MooncakeConnector 分布式，按需升级

如果你正被尾延迟折磨，PD 分离值得一试。如果你只是想跑更多请求，还是先把合体部署调好。
