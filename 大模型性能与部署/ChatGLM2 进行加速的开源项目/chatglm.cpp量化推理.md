# ChatGLM2 进行加速的开源项目

## 加速推理方案

- **fastllm**：全平台加速推理方案，单GPU批量推理每秒可达10000+token，手机端最低3G内存实时运行（骁龙865上约4~5 token/s）
- **chatglm.cpp**：类似 llama.cpp 的 CPU 量化加速推理方案，实现 Mac 笔记本上实时对话

---

## chatglm.cpp 量化推理

### 参考资料

- https://www.cnblogs.com/yjmyzz/p/chatglm_cpp.html
- https://cloud.tencent.com/developer/article/2336318

### ChatGLM3 量化：Python 和 C++ 两种实现方式

#### 第一步：安装依赖包

```bash
pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate modelscope
pip install bitsandbytes
pip install --upgrade accelerate
```

#### Python BitsAndBytes 4bit 量化

```python
from modelscope import AutoTokenizer, AutoModel, snapshot_download, BitsAndBytesConfig
import torch

model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="master")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, quantization_config=bnb_config)

# 使用量化后的模型进行提问
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
```

> 可在线运行的 notebook 链接：在 Kaggle 网站搜索 `chatglm3-cpp`

---

## C++ 推理（chatglm.cpp）

### 1. 克隆仓库

```bash
git clone --recursive https://github.com/li-plus/chatglm.cpp.git
```

![[file-20260508224746253.png]]

### 2. 进入项目目录

```bash
cd chatglm.cpp
```

### 3. 下载已转换的 GGML 模型

```bash
git clone https://www.modelscope.cn/tiansz/chatglm3-6b-ggml.git
```

或使用 ModelScope 库下载：

```python
from modelscope import AutoTokenizer, AutoModel, snapshot_download
model_dir = snapshot_download("tiansz/chatglm3-6b-ggml")
print(model_dir)
```

### 4. 编译项目

```bash
cmake -B build
cmake --build build -j --config Release
```

### 5. CPU 推理

```bash
./build/bin/main -m chatglm3-6b-ggml/chatglm3-ggml.bin -p 你好
```

### 6. GPU 推理

重新编译以启用 CUDA：

```bash
sudo apt-get --yes install build-essential
pip install --upgrade cmake
cmake -B build -DGGML_CUBLAS=ON && cmake --build build -j
```

编译完成后使用 GPU 推理，速度有大幅度提升：

```bash
./build/bin/main -m chatglm3-6b-ggml/chatglm3-ggml.bin -p 你好
```

### 7. Python 调用 C++ GPU 推理

安装 Python 包：

```bash
CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install -U chatglm-cpp
```

```bash
cd examples
```

```python
import chatglm_cpp
pipeline = chatglm_cpp.Pipeline("../chatglm3-6b-ggml/chatglm3-ggml.bin")
pipeline.chat(["你好"])
```

---

## 量化自己的微调模型转成 GGML 格式

![[file-20260508224746251.png]]

```bash
/home/qyc/chatglm.cpp# python3 chatglm_cpp/convert.py -i /home/qyc/LLaMA-Factory-main/save_custom/opt_matchprompt_sft_chatglm3_rola -t q4_0 -o chatglm-ggml.bin
```

---

## 交互式 CLI 命令体验模型

> 这种模式下，聊天记录会被带到下一次对话中。

```bash
./build/bin/main -m chatglm-ggml.bin -i -l 1024 --temp 0 -t 8
```

### 命令参数介绍

![[file-20260508224746248.png]]

![[大模型性能与部署/chatGLM的高性能推理库/assets/ChatGLM2 进行加速的开源项目/a3f0c9ca34532faef2e169728f6bde9e_MD5.png]]

---

## Python 调用 GGML 模型

### 安装

| 环境 | 安装命令 |
|------|---------|
| 纯 CPU | `pip install -U chatglm-cpp` |
| NVIDIA CUDA | `CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install -U chatglm-cpp` |

### 推理代码

```python
import chatglm_cpp
import time

pipeline = chatglm_cpp.Pipeline(r"/home/qyc/chatglm.cpp/chatglm3-6b-ggml/chatglm3-ggml.bin")

for i in range(0, 10):
    t1 = time.time()
    responses = pipeline.chat(["你好"], max_length=512, do_sample=True, temperature=0, num_threads=8)
    t2 = time.time()
    print(f"{t2 - t1}, responses={responses} {(t2-t1)/len_n*1000}ms/token")
input("!!!!!!!!!!")
```

> **注意**：当前目录下有一个目录名为 `chatglm_cpp`，与 import 的依赖同名。后续使用这个包都会出现冲突，需要把运行的脚本放到另外一个目录下运行，并注意加载的模型路径。或者在安装后把 `chatglm_cpp` 目录重命名，比如改为 `chatglm_cpp.origin`。

---

## 问题解决方案

### 1. cuBLAS 编译错误

> 错误信息：`Value 'sm_30' is not defined for option 'gpu-name'`

可能是因为电脑中有安装重复的 CUDA 工具包，需要卸载掉：

```bash
apt-cache policy nvidia-cuda-toolkit
sudo apt remove nvidia-cuda-toolkit
```

### 2. 提示没有 CMAKE_CUDA_ARCHITECTURES 参数

在命令行中添加 `-DCMAKE_CUDA_ARCHITECTURES` 参数，参数值是一个数字，具体的值需要上 NVIDIA 官网网址查找对应显卡架构。

### 3. 找不到对应的 CUDA 型号，或者 CUDA 版本不匹配

添加 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc` 参数，注意根据实际情况修改路径。
