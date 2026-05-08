代码仓库：https://github.com/mit-han-lab/streaming-llm
论文题目: Efficient Streaming Language Models with Attention Sinks


安装：
```
conda create -yn streaming python=3.8
conda activate streaming
pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece
python setup.py develop

```
#  conda deactivate
```
测试： python examples/run_streaming_llama.py  --enable_streaming  --model_name_or_path /home/qyc/bert/Llama2-Chinese-7b-Chat

```

原理简介：
目前的语言模型对输入长度有限制,超过预训练的文本长度就会性能下降。需要研究如何将语言模型应用到流式的长文本场景。本文提出的StreamingLLM可以解决这个问题。
[[大模型性能与部署/assets/streaming-llm推理突破长度限制/9c3b75309c408ec4bbb148dc69203582_MD5.png|Open: file-20260508222204740.png]]
![[大模型性能与部署/assets/streaming-llm推理突破长度限制/9c3b75309c408ec4bbb148dc69203582_MD5.png]]
(a) Dense Attention 需要缓存所有前序token的键值(KV)状态,计算复杂度为O(T^2),随着T增大,内存占用和延迟也线性增加。当文本长度超过预训练长度时,性能下降。
(b) Window Attention 只缓存最近L个token的KV状态,计算复杂度为O(TL),内存占用和延迟恒定。但是当文本长度超过缓存大小L时,由于丢弃了初始token的KV状态,性能急剧下降。
(c) Sliding Window with Re-computation 在生成每个新token时重新计算最近L个token的KV状态,计算复杂度为O(TL^2)。性能良好但计算非常慢。
(d) StreamingLLM 同时缓存注意力汇(几个初始token)和最近L个token的KV状态,计算复杂度为O(TL),既高效又保证了在超长文本上的稳定性能。



本文提出StreamingLLM框架,同时缓存attention sink(几个初始token)和最近的token,来实现对无限长文本的高效稳定语言建模,而无需微调模型。
具体来说,StreamingLLM中的attention计算只基于缓存中的相对位置编码,而不是原始文本中的绝对位置。


![[大模型性能与部署/assets/streaming-llm推理突破长度限制/10317ea0193d97087526b521ac21fd24_MD5.png]]
对于RoPE相对位置编码,在每个解码步骤,对缓存中的Keys应用旋转变换;对于ALiBi,则直接在缓存中应用连续的线性偏置。这种基于缓存的位置编码对StreamingLLM的有效性至关重要。



StreamingLLM的设计提供了依据:
窗口attention机制会在文本长度超过缓存大小时失败,因为它丢弃了初始token的键值状态。本文通过可视化attention分布,发现初始token获得了意外地高的attention分数,扮演“attention sink”的角色,
用于平衡softmax的归一化约束。这里的attention sink指的是模型中不具语义重要性但获得高attention分数的token。

![[大模型性能与部署/assets/streaming-llm推理突破长度限制/2e016732b73f983aa3b4452889ac9667_MD5.png|283]]
即使当前token的查询向量与上下文向量匹配性不强,模型仍需要将attention分配到一些token上以满足归一化约束。初始token由于对所有后续token可见,更容易在训练中学习成为attention sink。

![[大模型性能与部署/assets/streaming-llm推理突破长度限制/aed96f2fa5afa3a2b86afff4c0bf0170_MD5.png]]








