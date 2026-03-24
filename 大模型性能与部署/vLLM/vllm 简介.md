---
tags:
  - vllm
---
vLLM 用于大模型并行推理加速，核心是 PagedAttention 算法，官网为：https://vllm.ai/。

官网文档：https://vllm.readthedocs.io/en/latest/getting_started/installation.html
演示代碼：https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py

参考帖子：
https://zhuanlan.zhihu.com/p/649974825
https://zhuanlan.zhihu.com/p/691038809

安装：
pip intall vllm  # This may take 5-10 minutes.

容器安装：
请到https://hub.docker.com/搜索最新的vllm镜像
docker pull vllm/vllm-openai:v0.7.2

## vLLM 提供了丰富的 API 端点，支持以下接口：

| 端点                             | 说明            | 适用模型                     |
| ------------------------------ | ------------- | ------------------------ |
| **`/v1/chat/completions`**     | 对话补全（推荐）      | 带 chat template 的文本生成模型  |
| **`/v1/completions`**          | 文本补全（Legacy）  | 文本生成模型                   |
| **`/v1/responses`**            | Responses API | 文本生成模型                   |
| **`/v1/embeddings`**           | 文本嵌入          | Embedding 模型             |
| **`/v1/audio/transcriptions`** | 语音转文字         | ASR 模型（如 Whisper）\[ ^1^] |
| **`/v1/audio/translations`**   | 语音翻译          | ASR 模型                   |
| **`/v1/realtime`**             | 实时语音          | ASR 模型                   |
| **`/v1/models`**               | 列出可用模型        | -                        |
| **`/health`**                  | 健康检查          | -                        |


代码使用说明：
class LLM:
    这是一个名为LLM（语言模型）的Python类，这个类用于从给定的提示和采样参数生成文本。
    类的主要部分包括tokenizer（用于将输入文本分词）、语言模型（可能分布在多个GPU上执行）
    以及为中间状态分配的GPU内存空间（也被称为KV缓存）。给定一批提示和采样参数，
    该类将使用智能批处理机制和高效的内存管理从模型中生成文本。

    这个类设计用于离线推理。在线服务的话，应使用AsyncLLMEngine类。
    对于参数列表，可以参见EngineArgs。

    Args:
        model: HuggingFace Transformers模型的名称或路径.
        tokenizer: HuggingFace Transformers分词器的名称或路径。默认为None。.
        tokenizer_mode: 分词器模式。"auto"将使用快速分词器（如果可用），
        "slow"将总是使用慢速分词器。默认为"auto"。.
        trust_remote_code: 当下载模型和分词器时，是否信任远程代码
        （例如，来自HuggingFace的代码）。默认为False。
        tensor_parallel_size: 用于分布式执行的GPU数量，使用张量并行性。默认为1。
        dtype: 模型权重和激活的数据类型。目前，我们支持float32、float16和bfloat16。
        如果是auto，我们使用在模型配置文件中指定的torch_dtype属性。
        但是，如果配置中的torch_dtype是float32，我们将使用float16。默认为"auto"。
        seed: 初始化采样的随机数生成器的种子。默认为0。
  

 原理：
用于自回归生成的 KV cache 占大量显存，受OS中的虚拟内存和分页的思想启发，提出了该 attention 优化算法，可在不连续的显存空间存储连续的 key 和 value。
用于将每个序列的 KV cache 分块（blocks），每块包含固定数量的 token 的 key 和 value 张量。


KV cashe的介紹：
空间换时间思想: 生成第n個token時， 前面n-1個token的kv計算是重複的操作浪費了計算資源。


PagedAttention介紹：

在图中：
请求（request）可理解为操作系统中的一个进程
逻辑内存（logical KV blocks）可理解为操作系统中的虚拟内存，每个block类比于虚拟内存中的一个page。每个block的大小是固定的，在vLLM中默认大小为16，即可装16个token的K/V值
块表（block table）可理解为操作系统中的虚拟内存到物理内存的映射表
物理内存（physical KV blocks）可理解为操作系统中的物理内存，物理块在gpu显存上，每个block类比于虚拟内存中的一个page

LLM推理过程通常分为两个阶段：
prefill阶段 （把整段prompt喂给模型做forward计算）和
decode阶段 （我们根据prompt的prefill结果，一个token一个token地生成response。耗时一般是更大的。）

vLLM通过一种名为PagedAttention的技术，动态地为请求分配KV cache显存，提升显存利用率


PagedAttention
通过KV Cache的技术，我们已经可以极大地提升LLM地推理速度，但是现有的Cache仍存在一些问题，
- Large：对于LLaMA-13B中的单个序列，它占用高达1.7GB的内存。
- Dynamic：它的大小取决于序列长度，而序列长度具有高度可变和不可预测的特点。
因此，高效地管理KV Cache是一个重大挑战。现有系统（HuggingFace 默认实现是pytorch的内存分配策略）由于内存碎片化和过度预留而浪费了60%至80%的内存。

为了解决这个问题，我们引入了PagedAttention，这是一种受传统操作系统虚拟内存和分页概念启发的注意力算法。与传统的注意力算法不同，PagedAttention允许将连续的键和值存储在非连续的内存空间中。具体而言，PagedAttention将每个序列的KV缓存分成多个块，每个块包含固定数量的标记的键和值。在注意力计算过程中，PagedAttention Kernel高效地识别和获取这些块，采用并行的方式加速计算。（和ByteTransformer的思想有点像）

内存布局

由于块在内存中不需要连续存储，我们可以像操作系统的虚拟内存那样以更加灵活的方式管理键和值的缓存：可以将块看作页，标记看作字节，序列看作进程。序列的连续逻辑块通过块表映射到非连续的物理块。随着生成新的标记，序列的边长，物理块按需进行分配。

在PagedAttention中，内存浪费仅发生在序列的最后一个块中。这就使得我们的方案接近最优的内存使用率，仅有不到4%的浪费。通过内存效率的提升，我们能够显著提升BatchSize，同时进行多个序列的推理，提高GPU利用率，从而显著提高吞吐量。


注册自己模型的官网操作说明：
https://vllm.readthedocs.io/en/latest/models/adding_model.html


