---
tags:
  - onnx
  - netron
---

# Netron 与 ONNX 查看 ONNX 文件模型

## 一、ONNX 库加载与检查

### 1. 加载 ONNX 文件，检查文件完整性

```python
import onnx
# Load the ONNX model
model = onnx.load(onnx_model_file_path)
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
"""
graph tf2onnx (
  %Input-Token:0[FLOAT, unk__6532xunk__6533]
  %Input-Segment:0[FLOAT, unk__6534xunk__6535]
) initializers (
  %zero_reduce__698[FLOAT, scalar]
  %zero_const__753[INT32, 1]
  %slice_axes__899[INT32, 3]
  %slice_axes__765[INT32, 2]
...
  %Embedding-Token_1/Reshape_2:0 = Reshape(%Embedding-Token_1/MatMul:0, %Embedding-Token_1/Reshape_2__5418:0)
  %MLM-Bias/add:0 = Add(%Embedding-Token_1/Reshape_2:0, %MLM-Bias/Reshape__400)
  %MLM-Activation/Max:0 = GlobalMaxPool(%MLM-Bias/add:0)
  %MLM-Activation/sub:0 = Sub(%MLM-Bias/add:0, %MLM-Activation/Max:0)
  %MLM-Activation/Exp:0 = Exp(%MLM-Activation/sub:0)
  %MLM-Activation/Sum:0 = ReduceSum[axes = [2], keepdims = 1](%MLM-Activation/Exp:0)
  %cross_entropy_1/Identity:0 = Div(%MLM-Activation/Exp:0, %MLM-Activation/Sum:0)
  return %cross_entropy_1/Identity:0
}
"""
```

### 2. 查看模型输入输出信息
![[643b49841cc69f11a5c1ba4c6487fe2b_MD5.png]]
```python
MODEL = ort.InferenceSession(onnx_model_file_path,
                             providers=["CUDAExecutionProvider"])  # , providers=["CUDAExecutionProvider"]
print(f"加载onnx的模型文件: {MODEL}")
inputs_name = [x.name for x in MODEL.get_inputs()]
output_name = MODEL.get_outputs()[0].name
print(inputs_name) # ['Input-Token:0', 'Input-Segment:0']
print(output_name) # ['cross_entropy_1/Identity:0']
```

### 3. ONNX 模型简化

**安装 onnx-simplifier：**

```bash
pip install onnx-simplifier
```

```python
from onnxsim import simplify
import onnx
import json
input_path=  os.path.join(base_path, "save", f"{model_type}.onnx")
output_path= os.path.join(base_path, "save", f"{model_type}_simplify.onnx")
onnx_model = onnx.load(input_path)  # load onnx model


model_simp, check = simplify(onnx_model,input_shapes={'Input-Token:0': [1, 128], 'Input-Segment:0': [1, 128]})
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
```

---

## 二、Netron 库可视化

### 1. 安装

```bash
pip install netron
```

### 2. 加载 ONNX 实现可视化模型网络结构

```python
>>> import netron
>>> netron.start.__doc__
'Start serving model file at address and open in web browser.\n\n    Args:\n        file (string): Model file to serve.\n        log (bool, optional): Log details to console. Default: False\n        browse (bool, optional): Launch web browser, Default: True\n        address (tuple, optional): A (host, port) tuple, or a port number.\n\n    Returns:\n        A (host, port) address tuple.\n    '
>>> netron.start("/home/qyc/generateSurg/utils/etnet.onnx" , address=('192.168.0.5',8888))
Serving '/home/qyc/generateSurg/utils/etnet.onnx' at http://192.168.0.5:8888
('192.168.0.5', 8888)
```

### 3. 查看模型

打开网络查看模型结构：`http://192.168.0.5:8888/`

> 💡 **重点看输入输出的数据格式**
[[98a2dcf34a86dc5a1ef1aeb9750e7bb2_MD5.png|Open: file-20260508221355849.png]]
![[98a2dcf34a86dc5a1ef1aeb9750e7bb2_MD5.png]]