---
title: "大模型推理加速：看图学KV Cache"
source: "https://zhuanlan.zhihu.com/p/662498827"
author:
  - "[[看图学​​双一流计算机硕士｜前百度资深算法工程师｜多年互联网大厂经验]]"
published:
tags:
  - "clippings"
---
![[公众号文章/assets/大模型推理加速：看图学KV Cache/561ca89f8f1dc0e15823ad5e406ba427_MD5.png]]

大模型推理加速：看图学KV Cache

![[公众号文章/assets/大模型推理加速：看图学KV Cache/8dd61dd0324b5eb67d7c8c312481a68d_MD5.jpg]]

KV Cache 是 Transformer 标配的推理加速功能，transformer官方use\_cache这个参数默认是True，但是它 只能用于 Decoder架构 的模型 ，这是因为Decoder有 Causal Mask ，在推理的时候前面已经生成的字符不需要与后面的字符产生attention，从而使得前面已经计算的K和V可以缓存起来。

一些大模型的知识整理到下面两个目录了：
我们先看一下不使用KV Cache的推理过程。假设模型最终生成了“遥遥领先”4个字。 当模型生成第一个“遥”字时，input=" ", " "是 起始字符。 Attention 的计算如下： 为了看上去方便，我们 暂时忽略scale项 d ， 但是要注意这个scale面试时经常考。 如上图所示，最终Attention的计算公式如下，（softmaxed 表示已经按行进行了softmax）: A t 1 ( Q, K V ) = softmax T → softmaxed 当模型生成第二个“遥”字时，input=" 遥", Attention的计算如下： 当 变为矩阵时，softmax 会针对 行 进行计算。写详细一点如下，softmaxed 表示已经按行进行了softmax。 假设 表示 Attention 的第一行， 2 表示 Attention 的第二行，则根据上面推导， 其计算公式为： + 你会发现，由于 这个值会mask掉， 在第二步参与的计算与第一步是一样的，而且第二步生成的 也仅仅依赖于 ，与 毫无关系。 的计算也仅仅依赖于 当模型生成第三个“领”字时，input=" 遥遥"Attention的计算如下： 详细的推导参考第二步，其计算公式为： 3 同样的， k 只与 有关。 当模型生成第四个“先”字时，input=" 遥遥领"Attention的计算如下： 4 和之前类似，不再赘述。 看上面图和公式，我们可以得出结论： 当前计算方式存在大量冗余计算。 推理第 x 个字符的时候只需要输入字符 − 即可。 我们每一步其实之需要根据 计算 就可以，之前已经计算的Attention完全不需要重新计算。但是 和 是全程参与计算的，所以这里我们需要把每一步的 缓存起来。所以说叫KV Cache好像有点不太对，因为KV本来就需要全程计算，可能叫增量KV计算会更好理解。 下面4张图展示了使用KV Cache和不使用的对比。 下面是gpt里面KV Cache的实现。其实明白了原理后代码实现简单的不得了,就是concat操作而已。
github.com/huggingface/ 
```text
if layer_past is not None:
        past_key, past_value = layer_past
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
    
    if use_cache is True:
        present = (key, value)
    else:
        present = None
    
    if self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
    else:
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
```


最后需要注意当 sequence特别长的时候，KV Cache其实还是个 Memory刺客 。 比如batch\_size=32, head=32, layer=32, dim\_size=4096, seq\_length=2048, float32类型，则需要占用的显存为（感谢网友指正） 2 \* 32 \* 4096 \* 2048 \* 32 \* 4 / 1024/1024/1024 /1024 = 64G。 