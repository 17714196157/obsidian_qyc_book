Title: 大模型推理加速：看图学KV Cache

URL Source: https://zhuanlan.zhihu.com/p/662498827

Markdown Content:
KV Cache是Transformer标配的推理加速功能，transformer官方use\_cache这个参数默认是True，但是它**只能用于Decoder架构的模型**，这是因为Decoder有Causal Mask，在推理的时候前面已经生成的字符不需要与后面的字符产生attention，从而使得前面已经计算的K和V可以缓存起来。

我们先看一下不使用KV Cache的推理过程。假设模型最终生成了“遥遥领先”4个字。

当模型生成第一个“遥”字时，input="<s>", "<s>"是起始字符。Attention的计算如下：

![Image 1](https://pic2.zhimg.com/v2-f8706213a1f04fa1e41533bc0eeef601_b.jpg)

为了看上去方便，我们**暂时忽略scale项** d\\sqrt{d}， 但是要注意这个scale面试时经常考。

如上图所示，最终Attention的计算公式如下，（softmaxed 表示已经按行进行了softmax）:

Att1(Q,K,V)\=softmax(Q1K1T)V1→\=softmaxed(Q1K1T)V1→{\\color{orange}{Att\_1}}(Q, K, V) = \\text{softmax}({\\color{orange}{Q\_1}} K\_1^T) \\vec{V\_1} = \\text{softmaxed}({\\color{orange}{Q\_1}} K\_1^T) \\vec{V\_1} \\\\

当模型生成第二个“遥”字时，input="<s>遥", Attention的计算如下：

![Image 2](https://pic4.zhimg.com/v2-81197e811503d1ffa5f864f164127ddb_b.jpg)

当 QKTQK^T 变为矩阵时，softmax 会针对 **行** 进行计算。写详细一点如下，softmaxed 表示已经按行进行了softmax。

\\begin{aligned} {\\color{orange}{Att\_{step 2}}}(Q, K, V) &= \\text{softmax}(\\begin{bmatrix}{\\color{orange}{Q\_1}} K\_1^T & -\\infty \\\\{\\color{red}{Q\_2}} K\_1^T &{\\color{red}{Q\_2}} K\_2^T \\end{bmatrix}) \\begin{bmatrix} \\vec{V\_1} \\\\ \\vec{V\_2} \\end{bmatrix}\\\\ &= (\\begin{bmatrix}{\\text{softmaxed}(\\color{orange}{Q\_1}} K\_1^T) & \\text{softmaxed}(-\\infty) \\\\ {\\text{softmaxed}(\\color{red}{Q\_2}} K\_1^T) & \\text{softmaxed}({\\color{red}{Q\_2}} K\_2^T) \\end{bmatrix}) \\begin{bmatrix} \\vec{V\_1} \\\\ \\vec{V\_2} \\end{bmatrix} \\\\ &= (\\begin{bmatrix}{\\text{softmaxed}(\\color{orange}{Q\_1}} K\_1^T) & 0 \\\\ {\\text{softmaxed}(\\color{red}{Q\_2}} K\_1^T) & \\text{softmaxed}({\\color{red}{Q\_2}} K\_2^T) \\end{bmatrix}) \\begin{bmatrix} \\vec{V\_1} \\\\ \\vec{V\_2} \\end{bmatrix} \\\\ &= (\\begin{bmatrix}{\\text{softmaxed}(\\color{orange}{Q\_1}} K\_1^T) \\times \\vec{V1} + 0 \\times \\vec{V2} \\\\ {\\text{softmaxed}(\\color{red}{Q\_2}} K\_1^T) \\times \\vec{V1} + \\text{softmaxed}({\\color{red}{Q\_2} } K\_2^T) \\times \\vec{V2}\\end{bmatrix}) \\\\ &= (\\begin{bmatrix}{\\text{softmaxed}(\\color{orange}{Q\_1}} K\_1^T) \\times \\vec{V1} \\\\ {\\text{softmaxed}(\\color{red}{Q\_2}} K\_1^T) \\times \\vec{V1} + \\text{softmaxed}({\\color{red}{Q\_2} } K\_2^T) \\times \\vec{V2}\\end{bmatrix}) \\\\ \\end{aligned}

假设 {\\color{orange}{Att\_1}}(Q, K, V) 表示 Attention 的第一行， {\\color{orange}{Att\_2}}(Q, K, V) 表示 Attention 的第二行，则根据上面推导，

其计算公式为：

\\begin{aligned} {\\color{orange}{Att\_1}}(Q, K, V) &= \\text{softmaxed}({\\color{orange}{Q\_1}} K\_1^T) \\vec{V\_1} \\\\ {\\color{red}{Att\_2}}(Q, K, V) &= \\text{softmaxed}({\\color{red}{Q\_2}} K\_1^T) \\vec{V\_1} + \\text{softmaxed}({\\color{red}{Q\_2}} K\_2^T) \\vec{V\_2 } \\end{aligned} \\\\

你会发现，由于 Q\_1 K\_2^T 这个值会mask掉，

*   Q\_1 **在第二步参与的计算与第一步是一样的，而且第二步生成的** V\_1 **也仅仅依赖于** Q\_1 **，与** Q\_2 **毫无关系。**
*   V\_2 **的计算也仅仅依赖于** Q\_2 **，与** Q\_1 **毫无关系。**

当模型生成第三个“领”字时，input="<s>遥遥"Attention的计算如下：

![Image 3](https://pic1.zhimg.com/v2-29420723618e20a24dc3b6c329c570c8_b.jpg)

详细的推导参考第二步，其计算公式为：

\\begin{aligned} {\\color{orange}{Att\_1}}(Q, K, V) &= \\text{softmaxed}({\\color{orange}{Q\_1}} K\_1^T) \\vec{V\_1} \\\\ {\\color{red}{Att\_2}}(Q, K, V) &= \\text{softmaxed}({\\color{red}{Q\_2}} K\_1^T) \\vec{V\_1} + \\text{softmaxed}({\\color{red}{Q\_2}} K\_2^T) \\vec{V\_2 } \\\\ {\\color{purple}{Att\_3}}(Q, K, V) &= \\text{softmaxed}({\\color{purple}{Q\_3}} K\_1^T) \\vec{V\_1} + \\text{softmaxed}({\\color{purple}{Q\_3}} K\_2^T) \\vec{V\_2 } + \\text{softmaxed}({\\color{purple}{Q\_3}} K\_3^T) \\vec{V\_3 } \\end{aligned} \\\\

同样的， Att\_k 只与 Q\_k 有关。

当模型生成第四个“先”字时，input="<s>遥遥领"Attention的计算如下：

![Image 4](https://pic3.zhimg.com/v2-7bb8303b0a82b7ae668e2e9327b274e2_b.jpg)

\\begin{aligned} {\\color{orange}{Att\_1}}(Q, K, V) &= \\text{softmaxed}({\\color{orange}{Q\_1}} K\_1^T) \\vec{V\_1} \\\\ {\\color{red}{Att\_2}}(Q, K, V) &= \\text{softmaxed}({\\color{red}{Q\_2}} K\_1^T) \\vec{V\_1} + \\text{softmaxed}({\\color{red}{Q\_2}} K\_2^T) \\vec{V\_2 } \\\\ {\\color{purple}{Att\_3}}(Q, K, V) &= \\text{softmaxed}({\\color{purple}{Q\_3}} K\_1^T) \\vec{V\_1} + \\text{softmaxed}({\\color{purple}{Q\_3}} K\_2^T) \\vec{V\_2 } + \\text{softmaxed}({\\color{purple}{Q\_3}} K\_3^T) \\vec{V\_3 } \\\\ {\\color{brown}{Att\_4}}(Q, K, V) &= \\text{softmaxed}({\\color{brown}{Q\_4}} K\_1^T) \\vec{V\_1} + \\text{softmaxed}({\\color{brown}{Q\_4}} K\_2^T) \\vec{V\_2 } + \\text{softmaxed}({\\color{brown}{Q\_4}} K\_3^T) \\vec{V\_3 } + \\text{softmaxed}({\\color{brown}{Q\_4}} K\_4^T) \\vec{V\_4 } \\end{aligned} \\\\

和之前类似，不再赘述。

看上面图和公式，我们可以得出结论：

1.  **当前计算方式存在大量冗余计算。**
2.  Att\_k **只与** Q\_k **有关。**
3.  **推理第** x\_k **个字符的时候只需要输入字符** x\_{k-1}**即可。**

我们每一步其实之需要根据 Q\_k 计算 Att\_k 就可以，之前已经计算的Attention完全不需要重新计算。但是 K 和 V 是全程参与计算的，所以这里我们需要把每一步的 K,V 缓存起来。所以说叫KV Cache好像有点不太对，因为KV本来就需要全程计算，可能叫增量KV计算会更好理解。

下面4张图展示了使用KV Cache和不使用的对比。

![Image 5](https://pic2.zhimg.com/v2-655b95ebfb7808563bead28bc89bb459_b.jpg)

下面是gpt里面KV Cache的实现。其实明白了原理后代码实现简单的不得了,就是concat操作而已。

[https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling\_gpt2.py#L318C1-L331C97](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py%23L318C1-L331C97)

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

最后需要注意当**sequence特别长的时候，KV Cache其实还是个Memory刺客**。

比如batch\_size=32, head=32, layer=32, dim\_size=4096, seq\_length=2048, float32类型，则需要占用的显存为（感谢网友指正） 2 \* 32 \* 4096 \* 2048 \* 32 \* 4 / 1024/1024/1024 /1024 = 64G。

历史相关文章
------