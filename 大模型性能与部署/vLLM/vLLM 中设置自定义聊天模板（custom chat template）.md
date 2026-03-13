---
tags:
  - vllm
---
一）修改模型的 tokenizer_config.json
适合模型未内置模板或希望永久生效的情况。
打开模型的 tokenizer_config.json 文件。
添加或修改 chat_template 字段为模板字符串

```python
或者命令行指定jinja模版
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /home/qwq-32b-gptq-int4  \
--chat-template  /home/enable_think_mode.jinja  \
--served-model-name qwq-32b-int4  \
--tensor-parallel-size 2 \
--port 8101 \
--max-model-len 8000 \
--gpu-memory-utilization 0.8 \
--uvicorn-log-level info
```




二) 如何获取模板内容？
1） 从 LLaMA-Factory 提取模板（推荐）：
```python
from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/model")
template = TEMPLATES["qwen"]
template.fix_jinja_template(tokenizer)
print(tokenizer.chat_template)
```
2)
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 加载原始 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/qyc/qwq-32b-gptq-int4-nothink")
custom_template_nothink = """
{%- for message in messages %}
    {%- if message.role == "system" %}
        {{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- if content is defined %}
            {{- '<|im_start|>assistant\n' + content + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif message.role == "tool" %}
        {{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {%- if enable_think_mode == True %}
        {{- '<|im_start|>assistant\n' }}
    {%- elif enable_think_mode == False %}
        {{- '<|im_start|>assistant\n<think>直接回答。</think>' }}
    {%- endif %}
{%- endif %}
""".strip()

# 覆盖模板
print(tokenizer.chat_template)

tokenizer.chat_template = custom_template_nothink
from jinja2 import Template

Template(custom_template_nothink)  # 如果这里也抛同样错，就确认模板本身坏了

# 创建采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
# 准备消息
messages = [
{"role": "system", "content": "你是一个有帮助的助手。"},
{"role": "user", "content": "解释一下量子计算的基本概念。"}
]
# prompt = tokenizer.apply_chat_template(messages)
q = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_think_mode=False)

print(q)
```

三） 接口检查用了什么模板
```python
curl -X POST http://192.168.0.172:8101/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{
 
   "model": "qwq-32b-int4",
    "messages": [
 
      {"role": "user", "content": "你好"}
 
    ]
 ,
   "temperature": 0.3,
    "enable_think_mode": false
  }'
```
