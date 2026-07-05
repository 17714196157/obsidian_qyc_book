onnxruntime

官方文档： https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#install
版本对应关系：

| ONNX Runtime | CUDA    | cuDNN    | Notes                                                                                  |
|--------------|---------|----------|----------------------------------------------------------------------------------------|
| 1.10         | 11.4    | 8.2.4 (Linux)<br>8.2.2.26 (Windows) | libcudart 11.4.43<br>libcufft 10.5.2.100<br>libcurand 10.2.5.120<br>libcublasLt 11.6.1.51<br>libcublas 11.6.1.51<br>libcudnn 8.2.4 |
| 1.9          | 11.4    | 8.2.4 (Linux)<br>8.2.2.26 (Windows) | libcudart 11.4.43<br>libcufft 10.5.2.100<br>libcurand 10.2.5.120<br>libcublasLt 11.6.1.51<br>libcublas 11.6.1.51<br>libcudnn 8.2.4 |
| 1.8          | 11.0.3  | 8.0.4 (Linux)<br>8.0.2.39 (Windows) | libcudart 11.0.221<br>libcufft 10.2.1.245<br>libcurand 10.2.1.245<br>libcublasLt 11.2.0.252<br>libcublas 11.2.0.252<br>libcudnn 8.0.4 |
| 1.7          | 11.0.3  | 8.0.4 (Linux)<br>8.0.2.39 (Windows) | libcudart 11.0.221<br>libcufft 10.2.1.245<br>libcurand 10.2.1.245<br>libcublasLt 11.2.0.252<br>libcublas 11.2.0.252<br>libcudnn 8.0.4 |
| 1.5-1.6      | 10.2    | 8.0.3    | CUDA 11 can be built from source                                                      |
| 1.2-1.4      | 10.1    | 7.6.5    | Requires cublas10-10.2.1.243; cublas 10.1.x will not work                             |
| 1.0-1.1      | 10.0    | 7.6.4    | CUDA versions from 9.1 up to 10.1, and cuDNN versions from 7.1 up to 7.4 should also work with Visual Studio 2017 |

```
find . -name  "cuda"
tip：需要先安装onnxruntime，再安装onnxruntime-gpu，这样才能使用GPU，否则下面
print(ort.get_device())  #检测当前的硬件情况
```


##### onnx加速模型推斷
尝试将基于bert的模型 转onnx部署预测， 分析与直接使用gpu版的耗时情

- 举例1）基于bert的文本推断任务，输入文本长度为：60
直接加载torch模型版的预测耗时：0.00299
onnx部署GPU版本预测 耗时： 0.0219

- 举例2） 基于bert的手术生成模型 
输入文本为：周围神经缝合术
直接加载 tensorflow-GPU版本耗时 cost time=2.0827696323394775
onnx部署GPU版本预测 cost time=0.4580955505371094
onnx部署CPU版本预测 cost time=2.151688575744629
[[b3ca10034a50954d75ea9325197ef727_MD5.png|Open: file-20260508220143578.png]]
![[b3ca10034a50954d75ea9325197ef727_MD5.png]]

onnx导出时设置动态batch
![[05aa33871e66aea8d901152179a32f14_MD5.png]]