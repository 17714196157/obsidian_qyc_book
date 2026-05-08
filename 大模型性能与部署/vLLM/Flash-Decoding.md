Flash-Decoding 基于 FlashAttention，同时引入了一个新的并行维度：键值序列的长度。
代码仓库：Flash-decoding 可以在以下链接中找到：
FlashAttention 包，从 v2.2 开始：https://github.com/Dao-AILab/flash-attention/tree/main

Flash-Decoding 也在键和值之间并行化，代价是一个小的最终归约（reduction 步骤）。
Flash-Decoding 主要有三个工作步骤：
首先，将键 / 值分成更小的块；
使用 FlashAttention 并行计算查询与每个这些分块的注意力，为每行和每个分块额外写入一个标量值：注意力值的 log-sum-exp
最后，通过对所有分块进行归约来计算实际输出，使用 log-sum-exp 来调整每个分块的贡献。

这一切之所以可行，都是因为注意力 /softmax 可以进行迭代计算。
在 Flash-Decoding 中，它在两个级别上被使用：在分块内部（类似 FlashAttention），以及跨分块进行最终的归约计算。


![[大模型性能与部署/vLLM/assets/Flash-Decoding/9c331b0e3ed20e8ffafa131fffcea50f_MD5.png]]
![[大模型性能与部署/vLLM/assets/Flash-Decoding/da441ed74b77c13ab84f51bcd3c1639d_MD5.png]]
