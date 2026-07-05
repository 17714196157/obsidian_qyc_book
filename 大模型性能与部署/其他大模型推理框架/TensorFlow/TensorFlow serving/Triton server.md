官网查看：https://developer.nvidia.com/nvidia-triton-inference-server
代码仓库：https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md

getting_started 快速启动一个推理服务示例：
准备一个onnx模型文件
```python
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
import os
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer, AlbertConfig, AlbertForSequenceClassification
```

# load model
```python
base_path = r"C:\qyc\workcode\externalCauses"
pretrained = os.path.join(base_path,"bert","albert_chinese_small" )

tokenizer = BertTokenizer.from_pretrained(pretrained)
config = AlbertConfig.from_pretrained(
    pretrained,
    num_labels=8,
    problem_type="single_label_classification")

classifier = AlbertForSequenceClassification.from_pretrained(   # AutoModelForSequenceClassification
    pretrained,
    config=config,)

```
# 导出一个onnx格式的模型文件
```python
input_ids, attention_mask, token_type_ids = tokenizer("猜猜我是谁", max_length=64, truncation=True, return_tensors="pt", padding="max_length").values()
print(input_ids)
print(attention_mask)
print(token_type_ids)
out = classifier(input_ids, attention_mask, token_type_ids)
print(out)
"""
SequenceClassifierOutput(loss=None, logits=tensor([[ 0.2975, -0.0801, -0.3124,  0.1221,  0.1839, -0.1893,  0.2580, -0.1748]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""
torch.onnx.export(classifier,
                (input_ids, attention_mask, token_type_ids),
                f="model.onnx",
                export_params=True,
                opset_version=11,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["output"],
                dynamic_axes={"input_ids":{0: "batch_size"},
                              "attention_mask":{0: "batch_size"},
                              "token_type_ids":{0: "batch_size"}})
# ##
# opset_version**   ---导出版本,triton下使用15会报错
# input_names     ---指定输入名
# output_names    ---指定输出
# dynamic_axes**    ---动态batch，一次可以推理多条结果
# ###
```

下载容器 docker pull  nvcr.io/nvidia/tritonserver:21.03-py3

配置文件存放位置和定义配置文件
```bash
(base) ➜  models tree
.
└── bert_onnx
    ├── 1
    │   └── model.onnx
    └── config.pbtxt


cat config.pbtxt:
(base) ➜  bert_onnx cat config.pbtxt
name: "bert_onnx"              # 模型名称
platform: "onnxruntime_onnx"            # 推理类型
max_batch_size: 8                       # 动态batch限制
input [                                 # 输入
    {
        name: "input_ids"               # 输入名
        data_type: TYPE_INT64          # 数据类型
        dims: [64]                      # shape  最大文本序列长度
    },
    {
        name: "attention_mask"
        data_type: TYPE_INT64
        dims: [64]
    },
    {
        name: "token_type_ids"
        data_type: TYPE_INT64
        dims: [64]
    }
    ]
output [                                # 输出
    {
      name: "output"
      data_type: TYPE_FP32
      dims: [8]
    }
  ]
dynamic_batching {                      # 动态批处理
    preferred_batch_size: [ 1, 4 ],     # 批数量
    max_queue_delay_microseconds: 10   # 等待时间
  }

```

启动服务端：
```bash
docker run --rm -p8100:8000 -p8101:8001 -p8102:8002 -v /home/qyc/models:/models nvcr.io/nvidia/tritonserver:21.03-py3 tritonserver --model-repository=/models

```
启动日志如下：
```bash
=============================
== Triton Inference Server ==
=============================

NVIDIA Release 21.03 (build 20851776)

Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use Docker with NVIDIA Container Toolkit to start this container; see
   https://github.com/NVIDIA/nvidia-docker.

I0130 09:03:29.769291 1 libtorch.cc:940] TRITONBACKEND_Initialize: pytorch
I0130 09:03:29.769324 1 libtorch.cc:950] Triton TRITONBACKEND API version: 1.0
I0130 09:03:29.769344 1 libtorch.cc:956] 'pytorch' TRITONBACKEND API version: 1.0
2023-01-30 09:03:29.917823: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
I0130 09:03:29.956010 1 tensorflow.cc:1880] TRITONBACKEND_Initialize: tensorflow
I0130 09:03:29.956030 1 tensorflow.cc:1890] Triton TRITONBACKEND API version: 1.0
I0130 09:03:29.956034 1 tensorflow.cc:1896] 'tensorflow' TRITONBACKEND API version: 1.0
I0130 09:03:29.956037 1 tensorflow.cc:1920] backend configuration:
{}
I0130 09:03:29.957463 1 onnxruntime.cc:1728] TRITONBACKEND_Initialize: onnxruntime
I0130 09:03:29.957476 1 onnxruntime.cc:1738] Triton TRITONBACKEND API version: 1.0
I0130 09:03:29.957480 1 onnxruntime.cc:1744] 'onnxruntime' TRITONBACKEND API version: 1.0
I0130 09:03:29.980677 1 openvino.cc:1166] TRITONBACKEND_Initialize: openvino
I0130 09:03:29.980709 1 openvino.cc:1176] Triton TRITONBACKEND API version: 1.0
I0130 09:03:29.980714 1 openvino.cc:1182] 'openvino' TRITONBACKEND API version: 1.0
E0130 09:03:29.980809 1 pinned_memory_manager.cc:202] failed to allocate pinned system memory: CUDA driver version is insufficient for CUDA runtime version
I0130 09:03:29.981640 1 model_repository_manager.cc:1065] loading: bert_onnx:1
I0130 09:03:30.082337 1 onnxruntime.cc:1787] TRITONBACKEND_ModelInitialize: bert_onnx (version 1)
I0130 09:03:30.086157 1 onnxruntime.cc:1830] TRITONBACKEND_ModelInstanceInitialize: bert_onnx (CPU device 0)
WARNING: Since openmp is enabled in this build, this API cannot be used to configure intra op num threads. Please use the openmp environment variables to control the number of threads.
I0130 09:03:30.204717 1 model_repository_manager.cc:1239] successfully loaded 'bert_onnx' version 1
I0130 09:03:30.204882 1 server.cc:500]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I0130 09:03:30.204940 1 server.cc:527]
+-------------+-----------------------------------------------------------------+--------+
| Backend     | Path                                                            | Config |
+-------------+-----------------------------------------------------------------+--------+
| pytorch     | /opt/tritonserver/backends/pytorch/libtriton_pytorch.so         | {}     |
| tensorflow  | /opt/tritonserver/backends/tensorflow1/libtriton_tensorflow1.so | {}     |
| onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | {}     |
| openvino    | /opt/tritonserver/backends/openvino/libtriton_openvino.so       | {}     |
+-------------+-----------------------------------------------------------------+--------+

I0130 09:03:30.204977 1 server.cc:570]
+-----------+---------+--------+
| Model     | Version | Status |
+-----------+---------+--------+
| bert_onnx | 1       | READY  |
+-----------+---------+--------+

I0130 09:03:30.205095 1 tritonserver.cc:1658]
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                                              |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                                             |
| server_version                   | 2.8.0                                                                                                                                              |
| server_extensions                | classification sequence model_repository schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data statistics |
| model_repository_path[0]         | /models                                                                                                                                            |
| model_control_mode               | MODE_NONE                                                                                                                                          |
| strict_model_config              | 1                                                                                                                                                  |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                                          |
| min_supported_compute_capability | 6.0                                                                                                                                                |
| strict_readiness                 | 1                                                                                                                                                  |
| exit_timeout                     | 30                                                                                                                                                 |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+

I0130 09:03:30.206149 1 grpc_server.cc:3983] Started GRPCInferenceService at 0.0.0.0:8001
I0130 09:03:30.206395 1 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I0130 09:03:30.247878 1 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002


查看服务健康状态：
(base) ➜ curl -v localhost:8100/v2/health/ready
*   Trying 127.0.0.1...
* Connected to localhost (127.0.0.1) port 8100 (#0)
> GET /v2/health/ready HTTP/1.1
> Host: localhost:8100
> User-Agent: curl/7.47.0
> Accept: */*
>
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
<
* Connection #0 to host localhost left intact

```


客户端调用代码:
```python
import tritonclient.grpc as grpcclient
triton_client = grpcclient.InferenceServerClient(
                                                url="192.168.0.5:8101",   # ---端口
                                                verbose=True,           # ---日志
#                                                 ssl=False,
#                                                 root_certificates=None,
#                                                 private_key=None,
#                                                 certificate_chain=None,
#                                                 creds=None,
#                                                 keepalive_options=None,
#                                                 channel_args=None,
)
text_list = ["阿斯蒂芬你", "avfavf", "参谋阿福", "asnmdff"]
input_ids, attention_mask, token_type_ids = tokenizer(text_list, max_length=64, padding="max_length", truncation=True, return_tensors="np").values()
input_ids = input_ids.astype(np.int64)
attention_mask = attention_mask.astype(np.int64)
token_type_ids = token_type_ids.astype(np.int64)

model_name = "bert_onnx"

# Infer
inputs = []
outputs = []

# 要注意datatype与config文件不一样，需要查官网
inputs.append(grpcclient.InferInput(name='input_ids', shape=input_ids.shape, datatype="INT64"))
inputs.append(grpcclient.InferInput(name='attention_mask', shape=input_ids.shape, datatype="INT64"))
inputs.append(grpcclient.InferInput(name='token_type_ids', shape=input_ids.shape, datatype="INT64"))

# Initialize the data
inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(attention_mask)
inputs[2].set_data_from_numpy(token_type_ids)

outputs.append(grpcclient.InferRequestedOutput('output'))

# Test with outputs     ---第一种是有指定outputs
results = triton_client.infer(
                              model_name=model_name,    # ---模型名称
                              # model_version="1",      # ---请求变量
                              inputs=inputs,            # ---版本名称
                              outputs=outputs,          # ---请求输出对象的列表，可以为空
                              client_timeout=100,       # ---每个请求的超时值，以微秒为单位
                              headers={'test': '10'})    # ---允许请求占用的最大端到端时间，以秒为单位


# Get the output arrays from the results  ---把结果转为numpy格式输出
output0_data = results.as_numpy('output')

print(output0_data)
```



