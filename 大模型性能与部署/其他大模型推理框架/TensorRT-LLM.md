## 代码仓库

- https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/docs/source/installation.md
- https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0/examples/chatglm2-6b

---

## 安装与构建 TensorRT-LLM

### 1. 创建 Python 虚拟环境，更新 pip 源地址

```bash
conda create -yn streaming python=3.10
conda activate streaming
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### 2. 下载代码库

```bash
apt-get -y install git git-lfs

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

git submodule update --init --recursive
git lfs install
git lfs pull
```

### 3. 创建容器

```bash
make -C docker release_build
```

<details>
<summary>构建输出日志</summary>

```
Run 'do-release-upgrade' to upgrade to it.
22 updates could not be installed automatically. For more details,
see /var/log/unattended-upgrades/unattended-upgrades.log
Your Hardware Enablement Stack (HWE) is supported until April 2025.
Last login: Thu Nov 16 16:53:59 2023 from 127.0.0.1
➜  ~ screen -r RT
 => [internal] load build definition from Dockerfile.multi                                                              0.0s
 => => transferring dockerfile: 38B                                                                                     0.0s
 => [internal] load .dockerignore                                                                                       0.0s
 => => transferring context: 35B                                                                                        0.0s
 => [internal] load metadata for nvcr.io/nvidia/pytorch:23.08-py3                                                       1.7s
[+] Building 1545.8s (24/30)
[+] Building 2098.3s (24/30)
[+] Building 2098.6s (24/30)
[+] Building 2099.4s (24/30)
[+] Building 3631.0s (24/30)
[+] Building 3631.1s (24/30)
[+] Building 3632.0s (24/30)
[+] Building 3739.9s (24/30)
[+] Building 3740.1s (24/30)
[+] Building 5213.9s (27/30)
 => [internal] load build definition from Dockerfile.multi                                                              0.0s
 => => transferring dockerfile: 38B                                                                                     0.0s
 => [internal] load .dockerignore                                                                                       0.0s
 => => transferring context: 35B                                                                                        0.0s
 => [internal] load metadata for nvcr.io/nvidia/pytorch:23.08-py3                                                       1.7s
 => [internal] load build context                                                                                       1.0s
 => => transferring context: 647.19kB                                                                                   1.0s
 => [base 1/1] FROM nvcr.io/nvidia/pytorch:23.08-py3@sha256:12a39f22d6e3a3cfcb285a238b6219475181672ff41a557a75bdeeef6d630740  0.0s
 => CACHED [devel  1/10] COPY docker/common/install_base.sh install_base.sh                                             0.0s
 => CACHED [devel  2/10] RUN bash ./install_base.sh && rm install_base.sh                                               0.0s
 => CACHED [devel  3/10] COPY docker/common/install_cmake.sh install_cmake.sh                                           0.0s
 => CACHED [devel  4/10] RUN bash ./install_cmake.sh && rm install_cmake.sh                                             0.0s
 => CACHED [devel  5/10] COPY docker/common/install_tensorrt.sh install_tensorrt.sh                                     0.0s
 => CACHED [devel  6/10] RUN bash ./install_tensorrt.sh && rm install_tensorrt.sh                                       0.0s
 => CACHED [devel  7/10] COPY docker/common/install_polygraphy.sh install_polygraphy.sh                                 0.0s
 => CACHED [devel  8/10] RUN bash ./install_polygraphy.sh && rm install_polygraphy.sh                                   0.0s
 => CACHED [devel  9/10] COPY docker/common/install_pytorch.sh install_pytorch.sh                                       0.0s
 => CACHED [devel 10/10] RUN bash ./install_pytorch.sh skip && rm install_pytorch.sh                                    0.0s
 => CACHED [release 1/6] WORKDIR /app/tensorrt_llm                                                                      0.0s
 => CACHED [wheel 1/9] WORKDIR /src/tensorrt_llm                                                                        0.0s
 => CACHED [wheel 2/9] COPY benchmarks benchmarks                                                                       0.0s
 => CACHED [wheel 3/9] COPY cpp cpp                                                                                     0.0s
 => CACHED [wheel 4/9] COPY benchmarks benchmarks                                                                       0.0s
 => CACHED [wheel 5/9] COPY scripts scripts                                                                             0.0s
 => CACHED [wheel 6/9] COPY tensorrt_llm tensorrt_llm                                                                   0.0s
 => CACHED [wheel 7/9] COPY 3rdparty 3rdparty                                                                           0.0s
 => CACHED [wheel 8/9] COPY setup.py requirements.txt ./                                                                0.0s
 => [wheel 9/9] RUN python3 scripts/build_wheel.py --clean --trt_root /usr/local/tensorrt                            5091.5s
 => [release 2/6] COPY --from=wheel /src/tensorrt_llm/build/tensorrt_llm*.whl .                                         0.9s
 => [release 3/6] COPY --from=wheel /src/tensorrt_llm/cpp/include/ include/                                             0.0s
 => [release 4/6] RUN pip install tensorrt_llm*.whl && rm tensorrt_llm*.whl                                          1739.7s
 => [release 5/6] COPY README.md ./                                                                                     0.2s
 => [release 6/6] COPY examples examples                                                                                0.1s
 => exporting to image                                                                                                 42.3s
 => => exporting layers                                                                                                42.2s
 => => writing image sha256:cf2f4bec964c6c8096f625559943934b78855da63c26d38338b79c0f166ebad9                            0.0s
 => => naming to docker.io/tensorrt_llm/release:latest                                                                  0.0s
make: Leaving directory '/home/qyc/TensorRT-LLM/docker'
```

</details>

---

## 进入容器环境

```bash
docker run -dt --gpus all --name LLM -v "/home/qyc/bert:/bert" tensorrt_llm/release:latest
docker exec -it LLM /bin/bash
cd /app/tensorrt_llm
```

## Build TensorRT Engine(s)

> TensorRT-LLM builds TensorRT engine(s) after loaded the weight from HuggingFace pytorch Model. The `build.py` script requires a single GPU to build the TensorRT engine(s).

### ChatGLM2-6B

```bash
python3 build.py --model_dir=/bert/chatglm2-6b --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_weight_only --weight_only_precision --max_batch_size 1
```

![[file-20260705101859435.png]]
**注意事项 Tips:**
- 可以通过添加 `--use_weight_only` 启用 int8 weight-only 量化
- 可以通过添加 `--enable_context_fmha` 为 ChatGLM2-6B 启用 FMHA kernels
- 构建会使用几十个 G 的内存，这里用量化就是为了省内存
- 目前不支持 LoRA 这样合并的模型

---

## 运行服务

### ChatGLM2-6B 单 GPU 推理

> To run a TensorRT-LLM ChatGLM2-6B model on a single GPU, you can use Python:

```bash
# Run the ChatGLM2-6B model on a single GPU
python3 run.py

python run_qq.py --engine_dir=./trtModel/ --input_text="你好"
```

---

## 客户端请求代码示例

```python
import argparse
import json
import os
import re
import time
import torch
import transformers
import tensorrt_llm
from tensorrt_llm import runtime
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from build import get_engine_name  # isort:skip

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=128)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='trtModel')
    parser.add_argument('--input_text', type=str, default='续写：北京市教育资源丰富')
    parser.add_argument(
        '--input_tokens',
        type=str,
        help='CSV file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    return parser.parse_args()


def process_response(responseList):
    for i, response in enumerate(responseList):
        response = response.strip()
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0],
                              r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0],
                              r"%s\1" % item[1], response)

        responseList[i] = response
    return responseList


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('chatglm2-6b', dtype, world_size,
                                  runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/bert/chatglm2-6b", trust_remote_code=True)
    input_ids = None
    input_text = None
    if args.input_tokens is None:
        input_text = args.input_text
        input_ids = tokenizer(
            [input_text], return_tensors="pt",
            padding=True)['input_ids'].int().contiguous().cuda()
    else:
        input_ids = []
        with open(args.input_tokens) as f_in:
            for line in f_in:
                for e in line.strip().split(','):
                    input_ids.append(int(e))
        input_text = "<ids from file>"
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int32).cuda().unsqueeze(0)
    input_lengths = torch.tensor(
        [input_ids.size(1) for _ in range(input_ids.size(0))]).int().cuda()

    model_config = ModelConfig(model_name="chatglm6b",
                               num_heads=num_heads,
                               num_kv_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               dtype=dtype)
    sampling_config = SamplingConfig(end_id=2,
                                     pad_id=0,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     top_p=args.top_p)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = runtime.GenerationSession(model_config, engine_buffer,
                                        runtime_mapping)
    t1 = time.time()
    decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len)
    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    # [output_len, batch_size, beam_width] -> [batch_size, output_len, beam_width]
    output_ids = output_ids.squeeze(1)
    torch.cuda.synchronize()
    t2 = time.time()
    print(t2-t1)
    for i in range(len(output_ids.tolist())):
        output_ids = output_ids.tolist()[i][input_ids.size(1):]

        outputList = tokenizer.batch_decode(output_ids,
                                            skip_special_tokens=True)
        output_text = process_response(outputList)
        print(f'***************************************')
        print(f'Input --->\n {input_text}')
        print(f'Output --->\n {"".join(output_text)}')
        print(f'***************************************')

    print("Finished!")
```

---

## GPT2 模型使用示例

> GPT2 上 PyTorch 的模型使用示例

```bash
docker exec -it LLM /bin/bash
cd /app/tensorrt_llm/examples/gpt/
```

### 步骤 1: 转换权重

> Convert weights from HF Transformers to FT format

```bash
# 示例 1: QCModel
python3 hf_gpt_convert.py -i /bert/QCModel -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16

# 示例 2: Wenzhong_GPT2_110M_BertTokenizer_chinese
python3 hf_gpt_convert.py -i /bert/Wenzhong_GPT2_110M_BertTokenizer_chinese -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
```

```
root@0fee8684ed3d:/app/tensorrt_llm/examples/gpt/c-model/gpt2# pwd
/app/tensorrt_llm/examples/gpt/c-model/gpt2
```

### 步骤 2: 构建 Engine

> To build the TensorRT engine(s) needed to run the GPT model

```bash
# 示例 1
python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin --remove_input_padding

# 示例 2
python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin --remove_input_padding
```

<details>
<summary>构建输出示例</summary>

```
s tensor is marked as an output.
[11/16/2023-14:10:27] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 4390 MiB
[11/16/2023-14:10:27] [TRT-LLM] [I] Total time of building gpt_float16_tp1_rank0.engine: 00:01:13
[11/16/2023-14:10:27] [TRT-LLM] [I] Config saved to gpt_outputs/config.json.
[11/16/2023-14:10:27] [TRT-LLM] [I] Serializing engine to gpt_outputs/gpt_float16_tp1_rank0.engine...
[11/16/2023-14:10:28] [TRT-LLM] [I] Engine serialized. Total time: 00:00:00
[11/16/2023-14:10:28] [TRT-LLM] [I] Timing cache serialized to model.cache
[11/16/2023-14:10:28] [TRT-LLM] [I] Total time of building all 1 engines: 00:01:22
```

</details>

构建产物大小：

```
root@f1c872333f66:/app/tensorrt_llm/examples/gpt/gpt_outputs# ls -l
total 217132
-rw-r--r-- 1 root root      1435 Nov 16 14:10 config.json
-rw-r--r-- 1 root root 222336084 Nov 16 14:10 gpt_float16_tp1_rank0.engine

root@0fee8684ed3d:/app/tensorrt_llm/examples/gpt/gpt_outputs# du -sh *
4.0K    config.json
300M    gpt_float16_tp1_rank0.engine
```

### 步骤 3: 执行推理

```bash
# 示例 1
python3 run.py --max_output_len=128 --engine_dir=./gpt_outputs

# 示例 2
python3 run.py --max_output_len=8
```
