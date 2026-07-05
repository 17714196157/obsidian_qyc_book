
## fastllm 简介

- 纯 C++ 实现，便于跨平台移植，可以在安卓上直接编译
- ARM 平台支持 NEON 指令集加速，X86 平台支持 AVX 指令集加速，NVIDIA 平台支持 CUDA 加速，各个平台速度都很快
- 支持浮点模型（FP32）、半精度模型（FP16）、量化模型（INT8、INT4）加速
- 支持 Batch 速度优化
- 支持流式输出，很方便实现打字机效果
- 支持并发计算时动态拼 Batch
- 支持 Python 调用
- 目前支持 ChatGLM 模型，各种 LLaMA 模型（ALPACA、VICUNA 等），BAICHUAN 模型，MOSS 模型

**代码仓库**：https://github.com/ztxz16/fastllm/

---

## 安装

### 安装 cmake

```bash
sudo apt-get --yes install build-essential
pip install --upgrade cmake
```

### 编译 fastllm

```bash
cd fastllm
mkdir build
cd build
cmake .. -DUSE_CUDA=ON    # 如果不使用GPU编译，那么使用 cmake .. -DUSE_CUDA=OFF
make -j
cd tools && python setup.py install
```

### 编译报错解决

**报错信息**：

```
cmake .. -DUSE_CUDA=ON
CMake Error at ... lib/python3.8/site-packages/cmake/data/share/cmake-3.26/Modules/CMakeDetermineCUDACompiler.cmake:277
CMAKE_CUDA_ARCHITECTURES must be non-empty if set.
Call Stack (most recent call first):
CMakeLists.txt:39 (enable_language)
```

**解决方案**：

```bash
cmake .. -DUSE_CUDA=ON -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75
```

---

## ChatGLM 模型导出

> 默认脚本导出 ChatGLM2-6b 模型。需要先安装 ChatGLM-6B 环境。如果使用自己 finetune 的模型，需要修改 `chatglm_export.py` 文件中创建 tokenizer、model 的代码。

```bash
cd build

# 导出 float16 模型
python3 tools/chatglm_export.py chatglm2-6b-fp16.flm float16

# 导出 int8 模型
python3 tools/chatglm_export.py chatglm2-6b-int8.flm int8

# 导出 int4 模型
python3 tools/chatglm_export.py chatglm2-6b-int4.flm int4
```

---

## C++ 命令行程序

### 命令行聊天程序（支持打字机效果）

```bash
./main -p chatglm2-6b-fp16.flm
```

> [!example]- 运行效果
> ```
> AVX: ON
> AVX2: ON
> AARCH64: OFF
> Neon FP16: OFF
> Neon DOT: OFF
> Load (200 / 200)
> Warmup...
> finish.
> 欢迎使用 chatglm 模型. 输入内容对话，reset清空历史记录，stop退出程序.
> 用户: 你好
> chatglm:
>  你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
> ```

---

### 测试推理速度

可以使用 benchmark 程序进行测速，根据不同配置、不同输入，推理速度也会有一些差别。

#### Batch = 128

```bash
/home/qyc/fastllm/build# ./benchmark -p chatglm2-6b-fp16.flm -f ../example/benchmark/prompts/beijing.txt -b 128 -l 64
# 128个并行batch，输出长度限制在64
```

> [!example]- 运行效果
> ```
[ user: "<FLM_FIX_TOKEN_64795>
北京有什么景点？<FLM_FIX_TOKEN_64796>", model: "
 北京作为中国的首都，有着丰富的历史文化和风景名胜。以下是一些著名的景点：
> 1. 故宫博物院：位于北京市中心，是中国古代建筑之最，也是世界上保存最> 完整、规模最大的木质结构古建筑群。
> 2. 颐和园：位于北京西郊，是"]
> batch: 128
> prompt token number = 896
> prompt use 0.691100 s
> prompt speed = 1296.483887 tokens / s
> output 8064 tokens
> use 5.979875 s
> speed = 1348.523193 tokens / s
> ```


#### Batch = 1

```bash
(tt) (base) root@maizi:/home/qyc/fastllm/build# ./benchmark -p chatglm2-6b-fp16.flm -f ../example/benchmark/prompts/beijing.txt -b 1 -l 64
```

> [!example]- 运行效果
> ```
> [ user: "<FLM_FIX_TOKEN_64795>
> 北京有什么景点？<FLM_FIX_TOKEN_64796>", model: "
 > 北京作为中国的首都，有着丰富的历史文化和风景名胜。以下是一些著名的景点：
> 1. 故宫博物院：位于北京市中心，是中国古代建筑之最，也是世界上保存最完整、规模最大的木质结构古建筑群。
> 2. 颐和园：位于北京西郊，是"]
> batch: 1
> prompt token number = 7
> prompt use 0.069184 s
> prompt speed = 101.179466 tokens / s
> output 63 tokens
> use 3.441717 s
> speed = 18.304817 tokens / s
> ```

### 简易 WebUI

![[94086a27b21886d1bf05693b8dac40c1_MD5.png]]
> 使用流式输出 + 动态 batch，可多路并发访问

```bash
./webui -p chatglm2-6b-fp16.flm --port 3333
```

### API Server

```bash
./apiserver -p chatglm2-6b-fp16.flm --port 3333
```

---

## Python 调用

build 下面有第三方库：

![[a0dd080adf6a12d4d65e26120c082289_MD5.png]]
```python
from fastllm_pytools import llm
import time

def args_parser():
    parser = argparse.ArgumentParser(description='fastllm_chat_demo')
    parser.add_argument('-p', '--path', type=str, required=True, default='', help='模型文件的路径')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    model = llm.model(args.path)

    history = []
    print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("AI:", end="")
        curResponse = ""
        t1 = time.time()
        for response in model.stream_response(query, history=history):
            curResponse += response
            print(response, flush=True, end="")
        t2 = time.time()
        len_n = len(curResponse)
        print(f"{t2 - t1}, responses={curResponse} {(t2 - t1) / len_n}s/token")
        history.append((query, curResponse))
```

---

## pyfastllm 库的使用

> 简介：pyfastllm 是基于 fastllm 的 Python API 接口实现。

### 安装

#### 1. 下载 pybind11 C++ 依赖

```bash
git submodule init
git submodule update  # 下载pybind11依赖
```

或者：

```bash
pip install pybind11
# 将编译下载的第三方库拷贝过去
scp -r /home/qyc/fastllm/third_party/pybind11 build-py/third_party/
```

#### 2. C++ 手动编译

```bash
mkdir build-py
cd build-py
cmake .. -DUSE_CUDA=ON -DPY_API=ON
make -j

# 将编译得到的 so 文件放到脚本的同级目录下
cp /home/qyc/fastllm/build-py/pyfastllm.cpython-310-x86_64-linux-gnu.so pyfastllm/examples/
cd /home/qyc/fastllm/pyfastllm/examples
```

#### 3. 安装 py 库

**GPU 版本**：

```bash
cd pyfastllm/
python3 setup.py build
python3 setup.py install
```

**CPU 版本**：

```bash
cd pyfastllm/
export USE_CUDA=OFF
python3 setup.py build
python3 setup.py install
```

---

## 启动 Web 服务端

![[bd3ea19956c8f9e056a2e570c443c7ed_MD5.png]]
```bash
(tt) (base) root@maizi:/home/qyc/fastllm/pyfastllm/examples# pwd
/home/qyc/fastllm/pyfastllm/examples

python web_api.py -m 0 -p /home/qyc/fastllm/build/chatglm2-6b-fp16.flm --max_batch_size 16
```

> 备注：`chatglm2-6b-fp16.flm` 是通过 `python3 tools/chatglm_export.py chatglm2-6b-fp16.flm float16` 导出的 float16 模型。


> [!example]- 服务端日志
> ```
> call dynamic_batch_stream_func: running: False, prompt queue size: 0
> msg_dict size: 0
> ['你好', '北京有那些景点']  type:<class 'list'>
> call dynamic_batch_stream_func: running: False, prompt queue size: 0
> INFO:     127.0.0.1:58626 - "POST /api/batch_chat HTTP/1.1" 200 OK
> msg_dict size: 0
> ```

### 接口请求示例

```bash
curl -H "Content-Type: application/json" -X POST -d '{"prompts": ["你好","北京有那些景点"], "max_length": 100}' "http://localhost:8000/api/batch_chat"
```

> [!example]- 响应结果
> ```
> "(1/2)\n prompt: 你好 \n response: ，我是人工智能助手。很高兴为您服务！请问有什么问题我可以帮您解答？\n(2/2)\n prompt: 北京有那些景点 \n response: 值得一游?\n 寝食难安 \n 北京作为中国的首都，有着丰富的历史文化和众多著名的旅游景点。以下是一些值得一游的景点：\n\n1. 故宫博物院：位于北京市中心，是中国古代建筑之最，也是世界上保存最完整、规模最大的木质结构古建筑群。\n\n2. 颐和园：位于北京西郊，是清朝皇家园林，被誉为"皇家园林博物馆"。\n\n3. 天安门广场：位于北京市中心，是中国\n"
> ```



### fastllm部署lora微调后的模型
fastllm部署lora微调后的模型
1. 用微调大模型，使用LLaMA-Factory微调chatglm3-6b模型
```
accelerate launch  src/train_bash.py \
--stage sft \
--model_name_or_path /home/qyc/bert/chatglm3-6b  \
--do_train \
--dataset opt_split_scense_sft_train \
--template chatglm3 \
--finetuning_type lora \
--lora_target query_key_value \
--output_dir /home/qyc/LLaMA-Factory-main/save/path_to_sft_checkpoint \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 1000 \
--learning_rate 5e-4 \
--num_train_epochs 10.0 \
--plot_loss \
--overwrite_output_dir true \
--fp16

```
1. 使用LLaMA-Factory将lora参数与基础模型合并导出 
```
python src/export_model.py \
    --model_name_or_path /home/qyc/bert/chatglm3-6b  \
    --template chatglm3 \
    --finetuning_type lora \
    --checkpoint_dir  /home/qyc/LLaMA-Factory-main/save/path_to_sft_checkpoint \
    --export_dir /home/qyc/LLaMA-Factory-main/save_custom/opt_split_scense_sft_chatglm3_rola
```
1. 测试单个请求的准确性， 会发现chatglm3的代码中解析响应有些不一样，需要修改
![[1c361d790ad82246f1e70e527a3b2cb0_MD5.png]]

2. fastllm 转换模型格式为flm
	python3 tools/chatglm_export.py chatglm3-6b-opt-fp16.flm float16 
![[dd9d4ffad44fc4fc6fc4d7c5b02103e7_MD5.png]]
2. 部署flm模型
启动web服务端：
![[bd3ea19956c8f9e056a2e570c443c7ed_MD5.png]]

```
(tt) (base) root@maizi:/home/qyc/fastllm/pyfastllm/examples# pwd
/home/qyc/fastllm/pyfastllm/examples
python web_api.py -m 0 -p /home/qyc/fastllm/build/chatglm3-6b-opt-fp16.flm  --max_batch_size 16

 curl -H "Content-Type: application/json"   -X POST -d '{"prompts": ["你好","北京有那些景点"], "max_length": 100}' "http://localhost:3333/api/batch_chat"   

ab -c 10 -n 100 -T application/json -p q.json http://localhost:3333/api/batch_chat 


```
