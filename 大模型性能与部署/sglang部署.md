## 官方资源

- **官方文档**：https://docs.sglang.ai/start/install.html
- **代码仓库**：https://github.com/sgl-project/sglang

---

## 安装

### 方式 1：源码安装

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

> **注意**：这里隐藏着一个官方 bug，目前这个工具兼容的 transformers 版本是 `4.48.3`。

```bash
pip install transformers==4.48.3 --force-reinstall --no-deps --index-url https://pypi.org/simple/ --extra-index-url https://download.pytorch.org/whl/cu124
```

安装完之后，验证版本：

```bash
python -c "import transformers; print(transformers.version)"
```

输出结果是 `4.48.3` 就没有问题了。

#### 正常模型加载

```bash
python -m sglang.launch_server --model-path /home/qyc/bert/DeepSeek-R1-Distill-Qwen-32B --host 0.0.0.0 --trust-remote-code --tp 2 --cuda-graph-max-bs 2 --max-total-tokens 2096
```

#### GGUF 量化

```bash
python -m sglang.launch_server --model-path /home/qyc/bert/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --quantization gguf --max-total-tokens 3096 --context-length 1024 --host 0.0.0.0 --trust-remote-code --tp 2
```

#### AWQ 量化

```bash
python -m sglang.launch_server --model-path /new_data/yangxuan/premodel/Qwen/QwQ-32B-AWQ --quantization awq --max-total-tokens 4096 --context-length 2048 --host 0.0.0.0 --trust-remote-code --tp 2
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `--cpu-offload-gb 4` | 卸载到 CPU 的显存大小，适用于显存有限的设备 |
| `--quantization gguf` | 指定量化格式 |
| `--max-total-tokens` | 最大文本长度 |
| `--context-length` | 设置上下文长度 |

---

### 方式 2：使用容器

```bash
docker run --gpus all -p 30000:30000 -v /home/qyc/bert:/models lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path models/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --port 30000
```

---

## 接口请求

### 方式 1：URL 命令请求

**Endpoint**：`http://192.168.0.181:30000/v1/chat/completions`

```json
{
  "model": "/home/qyc/bert/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "1+1说多少"}
  ]
}
```

### 方式 2：Requests 请求（HTTP）

```python
# coding:utf-8
import requests
url = f"http://192.168.0.181:30000/v1/chat/completions"
data = {
    "model": "/home/qyc/bert/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": ""}],
}
text = """
你是一名ICD编码专家。请严格按照提示，仔细分析手术操作过程报告内容，判断该病案报告内容更符合所列出的"可选的手术名称"中的哪些，并从中选择，而不是生成。
提示：
1、需要先分析报告内容中针对什么部位做了手术，再去判断手术的入路与具体的手术关键操作的描述。
2、结合手术部位、术式、入路,选择最合理的手术，如果没有合理的手术就返回空
3、输出的手术名称分别放在<answer>...</answer>中


可选的手术名称:经尿道输尿管镜输尿管异物取出术、经尿道输尿管镜肾盂异物取出术、经尿道输尿管镜输尿管取石术、经尿道输尿管镜肾盂取石术、经尿道输尿管镜输尿管激光碎石术、经尿道输尿管镜肾盂激光碎石术、经尿道输尿管镜输尿管气压弹道碎石术、经尿道输尿管镜肾盂气压弹道碎石术、经尿道输尿管镜输尿管超声碎石术、经尿道输尿管镜肾盂超声碎石术、经尿道输尿管镜输尿管激光碎石取石术、经尿道输尿管镜肾盂激光碎石取石术、经尿道输尿管镜输尿管气压弹道碎石取石术、经尿道输尿管镜肾盂气压弹道碎石取石术、经尿道输尿管镜输尿管超声碎石取石术、经尿道输尿管镜肾盂超声碎石取石术、经尿道输尿管/肾盂异物取出术、经尿道输尿管/肾盂取石术、经尿道输尿管/肾盂激光碎石术、经尿道输尿管/肾盂气压弹道碎石术、经尿道输尿管/肾盂超声碎石术、经尿道输尿管/肾盂激光碎石取石术、经尿道输尿管/肾盂气压弹道碎石取石术、经尿道输尿管/肾盂超声碎石取石术

手术操作过程内容:1.患者麻醉成功后，取截石位，手术区域消毒铺无菌巾。
2.输尿管镜进镜顺利，探查见尿道未见异常，膀胱内三角区及其余各壁未见异常，双侧输尿管开口清晰。沿右侧输尿管开口置入非血管导丝，右侧输尿管开口狭窄，予扩张后输尿管镜进入顺利，上行至输尿管上段见结石一枚，直径0.8cm左右，周围炎性息肉包裹明显，将结石推入肾盂，退出输尿管镜，沿非血管导丝置入输尿管负吸导引鞘，决定行输尿管镜下钬激光碎石术。
3.置入钬激光将结石击碎至0.1－0.2cm大小，置入三角网篮，取出结石，置入导丝，引导下留置双j管，退出器械，留置16号双腔导尿管。
4.术毕，安返病房。
"""
data["messages"][-1]["content"] = text
response = requests.post(url, json=data)
print(response.json())
```

<details>
<summary>响应结果示例</summary>

```python
{
    'id': 'af41f1c33078452882c8728a4e7db277',
    'object': 'chat.completion',
    'created': 1741055800,
    'model': '/home/qyc/bert/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf',
    'choices': [{
        'index': 0,
        'message': {
            'role': 'assistant',
            'content': ' 患者接受的是经尿道输尿管镜钬激光碎石术，取出输尿管上段的结石。因此，对应的是"经尿道输尿管镜输尿管激光碎石术"。不过，报告中提到将结石推入肾盂，可能涉及肾盂部分，但主要手术部位是输尿管。此外，还进行了碎石和取石操作，因此可能更准确的是"经尿道输尿管镜输尿管激光碎石取石术"。\n\n分析：手术中使用了输尿管镜，进入输尿管，发现结石，使用激光碎石，并取出结石。这一过程符合"经尿道输尿管镜输尿管激光碎石取石术"的描述。\n\n\n<answer>经尿道输尿管镜输尿管激光碎石取石术</answer>',
            'tool_calls': None
        },
        'logprobs': None,
        'finish_reason': 'stop',
        'matched_stop': 151643
    }],
    'usage': {
        'prompt_tokens': 745,
        'total_tokens': 929,
        'completion_tokens': 184,
        'prompt_tokens_details': None
    }
}
```

</details>
