---
tags:
  - vllm
---
vllm 客户端代码
官网： https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server
代码demo：https://github.com/vllm-project/vllm/blob/main/examples/api_client.py

### vLLM 提供了丰富的 API 端点，支持以下接口：

| 端点                             | 说明            | 适用模型                     |
| ------------------------------ | ------------- | ------------------------ |
| **`/v1/chat/completions`**     | 对话补全（推荐）      | 带 chat template 的文本生成模型  |
| **`/v1/completions`**          | 文本补全（Legacy）  | 文本生成模型                   |
| **`/v1/responses`**            | Responses API | 文本生成模型                   |
| **`/v1/embeddings`**           | 文本嵌入          | Embedding 模型             |
| **`/v1/audio/transcriptions`** | 语音转文字         | ASR 模型（如 Whisper）\[ ^1^] |
| **`/v1/audio/translations`**   | 语音翻译          | ASR 模型                   |
| **`/v1/realtime`**             | 实时语音          | ASR 模型                   |
| **`/v1/models`**               | 列出可用模型        | -                        |
| **`/health`**                  | 健康检查          | -                        |

**查看vllm现在加载了那些模型**
```python
http://192.168.0.172:8000/v1/models
{"object":"list","data":[{"id":"qwq-32b-gptq-int4","object":"model","created":1749107065,"owned_by":"vllm","root":"/home/qyc/bert/qwq-32b-gptq-int4","parent":null,"max_model_len":8000,"permission":[{"id":"modelperm-2e9509e6f6b6448da2ac11d2051fef20","object":"model_permission","created":1749107065,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}

```


### vllm部署多个模型方案：
1. vllm同时启动多个模型，当显存足够时
通过 --model 参数支持 多模型同时加载，可在单服务实例中托管多个模型
https://blog.csdn.net/su_xiao_wei/article/details/146425845

2. 利用Nginx转发给多个vllm服务
https://mcp.csdn.net/6835655f606a8318e85a85c0.html


一） 代码请求
示例1）vllm加速facebook/opt-125m
```python
from vllm import LLM, SamplingParams
from huggingface_hub import login
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# login("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", add_to_git_credential=True) # 查看 https://huggingface.co/settings/tokens

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

llm = LLM(model=r"/home/qyc/bert/facebook_opt_125m", trust_remote_code=True)  # facebook/opt-125m
outputs = llm.generate("tell me the difference between", sampling_params)

t1 = time.time()
outputs = llm.generate("tell me the difference between", sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(len(generated_text))
t2 = time.time()
print(f"LLM time cost: {t2-t1}")
"""
Prompt: 'tell me the difference between', Generated text: ' this and the DTS-HD 4.1/4.2 version.\nthere is no difference between the two, the dTS-HD 4.1 gives you a bit more room to record with\nYeah, good point.'
151
LLM time cost: 0.375929594039917
"""

from transformers import pipeline
import time
from transformers import AutoModelForCausalLM,AutoTokenizer

model = AutoModelForCausalLM.from_pretrained((r"/home/qyc/bert/facebook_opt_125m"))
tokenizer = AutoTokenizer.from_pretrained((r"/home/qyc/bert/facebook_opt_125m"))
generator = pipeline('text-generation', model= model, tokenizer=tokenizer)

generator("tell me the difference between", top_p=0.95, max_length=50, temperature=0.8)

t1 = time.time()
outputs = generator("tell me the difference between", top_p=0.95, max_length=50, temperature=0.8)
for output in outputs:
    generated_text = output["generated_text"]
    print(f"Generated text: {generated_text!r}")
    print(len(generated_text))
t2 = time.time()

print(f"hugingface time cost: {t2-t1}")
"""
Generated text: 'tell me the difference between a "good" and "bad" game.\nI\'m not sure what you mean by "good" but I\'m sure you mean "bad" in the sense that you\'re not playing a game that is "good'
178
hugingface time cost: 1.6622192859649658
"""

示例2）vllm加速GPT2 

from vllm import LLM, SamplingParams
from huggingface_hub import login
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

prompt_text = "离院方式->患者家属要求回当地，建议继续针对以上问题给予治疗。患者因病情变化、治疗方案改变中止退出临床路径"
model_path = r"/home/qyc/GPT2_work/model/epoch9_work"


# login("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", add_to_git_credential=True) # 查看 https://huggingface.co/settings/tokens

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, stop=["[PAD]", "[SEP]", "[CLS]"])

llm = LLM(model=model_path, trust_remote_code=True)  # facebook/opt-125m
outputs = llm.generate(prompt_text, sampling_params)

t1 = time.time()
outputs = llm.generate(prompt_text, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(len(generated_text))
t2 = time.time()
print(f"LLM time cost: {t2-t1}")
"""
Prompt: '离院方式->患者家属要求回当地，建议继续针对以上问题给予治疗。患者因病情变化、治疗方案改变中止退出临床路径', Generated text: '患 者 要 求 离 院'
vLLM time cost: 0.0435791015625
hugingface time cost: 0.23979997634887695
"""

from transformers import pipeline
import time
from transformers import GPT2LMHeadModel,AutoTokenizer

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
generator = pipeline('text-generation', model= model, tokenizer=tokenizer)

generator(prompt_text, top_p=0.95, max_length=50, temperature=0.8)

t1 = time.time()
outputs = generator(prompt_text, top_p=0.95, max_length=len(prompt_text)+1 + 6, temperature=0.8)
for output in outputs:
    generated_text = output["generated_text"]
    print(f"Generated text: {generated_text!r}")
    print(len(generated_text))
t2 = time.time()

print(f"hugingface time cost: {t2-t1}")
# Generated text: '离院方式->患者家属要求回当地，建议继续针对以上问题给予治疗。患者因病情变化、治疗方案改变中止退出临床路径 患 者 要 求 离 院'
# hugingface time cost: 0.23979997634887695
```


二） vllm.entrypoints.openai.api_server 命令行启动
```python
支持参数说明：
usage: api_server.py [-h] [--host HOST] [--port PORT] [--allow-credentials] [--allowed-origins ALLOWED_ORIGINS] [--allowed-methods ALLOWED_METHODS] [--allowed-headers ALLOWED_HEADERS]
                     [--api-key API_KEY] [--served-model-name SERVED_MODEL_NAME] [--lora-modules LORA_MODULES [LORA_MODULES ...]] [--chat-template CHAT_TEMPLATE]
                     [--response-role RESPONSE_ROLE] [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE] [--root-path ROOT_PATH] [--middleware MIDDLEWARE] [--model MODEL]
                     [--tokenizer TOKENIZER] [--revision REVISION] [--code-revision CODE_REVISION] [--tokenizer-revision TOKENIZER_REVISION] [--tokenizer-mode {auto,slow}]
                     [--trust-remote-code] [--download-dir DOWNLOAD_DIR] [--load-format {auto,pt,safetensors,npcache,dummy}] [--dtype {auto,half,float16,bfloat16,float,float32}]
                     [--kv-cache-dtype {auto,fp8_e5m2}] [--max-model-len MAX_MODEL_LEN] [--worker-use-ray] [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
                     [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS] [--block-size {8,16,32}] [--seed SEED]
                     [--swap-space SWAP_SPACE] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION] [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS] [--max-num-seqs MAX_NUM_SEQS]
                     [--max-paddings MAX_PADDINGS] [--disable-log-stats] [--quantization {awq,gptq,squeezellm,None}] [--enforce-eager]
                     [--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE] [--disable-custom-all-reduce] [--enable-lora] [--max-loras MAX_LORAS] [--max-lora-rank MAX_LORA_RANK]
                     [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE] [--lora-dtype {auto,float16,bfloat16,float32}] [--max-cpu-loras MAX_CPU_LORAS] [--device {cuda}] [--engine-use-ray]
                     [--disable-log-requests] [--max-log-len MAX_LOG_LEN]


--tensor-parallel-size  表示用两个gpu去加载模型  
--gpu-memory-utilization 0.1 表示占用显存百分比
```

实例1） chatglm3模型加速
启动服务 python -m vllm.entrypoints.openai.api_server --model "/home/qyc/bert/chatglm3-6b" --trust-remote-code --dtype float16 --tensor-parallel-size 2

本地模型调用：
```python
openai调用方式：
from openai import OpenAI
# 启动vllm的openai兼容server：
# export VLLM_USE_MODELSCOPE=True
# python -m vllm.entrypoints.openai.api_server --model 'qwen/Qwen-7B-Chat-Int4' --trust-remote-code -q gptq --dtype float16 --gpu-memory-utilization 0.6
# python -m vllm.entrypoints.openai.api_server --model '/home/qyc/bert/Qwen-7B-Chat-Int4' --trust-remote-code -q gptq --dtype float16
# 用vllm部署openai兼容的服务端接口，然后走ChatOpenAI客户端调用
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="/home/qyc/bert/chatglm3-6b",
                                      prompt="你好")
print("Completion result:", completion)
####################################################
langchain调用方式：
# 启动vllm的openai兼容server：
# export VLLM_USE_MODELSCOPE=True
# python -m vllm.entrypoints.openai.api_server --model 'qwen/Qwen-7B-Chat-Int4' --trust-remote-code -q gptq --dtype float16 --gpu-memory-utilization 0.6
# python -m vllm.entrypoints.openai.api_server --model '/home/qyc/bert/Qwen-7B-Chat-Int4' --trust-remote-code -q gptq --dtype float16
# 用vllm部署openai兼容的服务端接口，然后走ChatOpenAI客户端调用
import os
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
model_name = "/home/qyc/bert/chatglm3-6b"  # "qwen/Qwen-7B-Chat-Int4"  "/home/qyc/bert/chatglm3-6b"
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(
    openai_api_base=openai_api_base,
    openai_api_key=openai_api_key,
    model=model_name
)
print(chat.invoke("你好"))

curl   http://localhost:8000/v1/completions  -H "Content-Type: application/json"     -d '{
        "model": "/home/qyc/bert/chatglm3-6b",
        "prompt": ["你好","北京有哪些景点"],
        "max_tokens": 128,
        "temperature": 0.1
    }'
加速时：
LLM time cost: 1s
pytoch：
time cost：2.1s
###############################
ab并行请求测试：
(base) root@maizi:~# ab -c 10 -n 1000 -T application/json -p company.json http://localhost:8000/v1/completions 
Time taken for tests: 180.256 seconds 
c
(base) root@maizi:~# ab -c 1 -n 1000 -T application/json -p company.json http://localhost:8000/v1/completions
Time taken for tests: 973.089 second 
Time per request: 973.089 [ms] (mean)
```

```python
实例2） gpt2模型加速
curl -w %{time_namelookup}::%{time_total}"\n"   http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/qyc/GPT2_work/model/epoch9_work",
        "prompt": "离院方式->出院前病程记录 \r\n日期 2023-03-11 21:05  \r\n今晨查房，患者一般状况可，无发热畏寒、无恶心呕吐、无胸闷气喘、无腹痛腹胀、无明显阴道流血，饮食睡眠可，尿管在位畅，尿色清。左侧腹腔引流畅，引流出少量淡黄色液体。查体：神清、精神可，心肺查体无明显异常，全腹软，无压痛及反跳痛，腹部切口愈合良好，敷料干燥无渗血渗液，双下肢无水肿。沈杨主任医师查房后示：患者现一般情况可，建议患者继续抗感染治疗，患者要求回家休养，告知其可能存在感染加重、低氧血症、急性呼吸窘迫甚至感染性休克等风险，予办理出院。-\r\n                                                             /杨柳青\r\n\r\n",
        "max_tokens": 11,
        "temperature": 0.1
    }'
加速时：
LLM time cost: 0.036643028259277344
pytoch：
time cost：0.10579800605773926
###############################
ab并行请求测试：
(base) root@maizi:~# ab -c 10 -n 1000 -T application/json -p company.json
Time taken for tests: 13.933 seconds Time per request: 139.328 [ms] (mean)
(base) root@maizi:~# ab -c 1 -n 1000 -T application/json -p company.json
Time taken for tests: 61.852 seconds Time per request: 61.852 [ms] (mean)
```

```python
实例3） 启动 gemma 模型
 python -m vllm.entrypoints.openai.api_server --model  "/home/qyc/bert/gemma-7b-it"  --trust-remote-code  --dtype float16  --tensor-parallel-size 2
```

```python
实例4） 启动 Llama3 模型
python -m vllm.entrypoints.openai.api_server --model "/home/qyc/bert/Meta-Llama-3-8B-Instruct" --trust-remote-code --dtype float16 --tensor-parallel-size 2

from langchain.llms import OpenAI
# pass your api key
os.environ["OPENAI_API_KEY"] = "/home/qyc/bert/Meta-Llama-3-8B-Instruct"
openai_api_key = "EMPTY"
openai_api_base = "http://192.168.0.181:8000/v1"
llm = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,model_name="/home/qyc/bert/Meta-Llama-3-8B-Instruct"
)
completion = llm.generate(["你好"])
print("Completion result:", completion)
```

```python
实例5） 启动 glm4 模型
python -m vllm.entrypoints.openai.api_server --model "/home/qyc/bert/glm-4-9b-chat" --trust-remote-code --dtype float16 --tensor-parallel-size 2  --max-model-len 8192
# max-model-len 表示模型最大长度，限制可以节省启动需要的显存

接口情况麻溜：
 curl -s http://192.168.0.181:8000/v1/chat/completions     -H "Content-Type: application/json"     -d '{
    "model": "/home/qyc/bert/glm-4-9b-chat",
    "messages": [
      {
        "role": "user",
        "content": "Compose a poem that explains the concept of recursion in programming."
      }
    ]
  }'|jq


代码请求：
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
model_name = "/home/qyc/bert/glm-4-9b-chat"  # "qwen/Qwen-7B-Chat-Int4"  "/home/qyc/bert/glm-4-9b-chat" "/home/qyc/bert/Meta-Llama-3-8B-Instruct"
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
# # 请求秘钥
# openai_api_key = 'sk-cd55c3aa0aeb4cc280e64cc8954e82fb'
# openai_api_base="https://api.deepseek.com"
# model_name = "deepseek-chat" 

from langchain.chat_models import ChatOpenAI
model = ChatOpenAI(
    openai_api_base=openai_api_base,
    openai_api_key=openai_api_key,
    model=model_name,
    max_tokens=512,  # 最大生成token数量
    stop=["<|endoftext|>", "<|user|>", "<|observation|>"]  # chatglm3-6b 用 <|im_end|>， llama3-6b 用 <|eot_id|> ，glm-4-9b 用 "<|endoftext|>","<|user|>","<|observation|>"
)

```
 三） vllm.entrypoints.api_server 命令行启动


三） vllm serve 命令行启动
魔塔社区模型下载 qwq-32b 量化版本模型文件
```python
from modelscope import snapshot_download
model_dir = snapshot_download('tclf90/qwq-32b-gptq-int4',revision='g128v3')
print(model_dir)



CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /home/qyc/bert/qwq-32b-gptq-int4   \
--served-model-name qwq-32b-int4  \
--tensor-parallel-size 2 \
--port 8101 \
--max-model-len 8000 \
--gpu-memory-utilization 0.8 \
--uvicorn-log-level info

curl --location 'http://192.168.0.172:8101/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "qwq-32b-int4",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ]
  }'
```

四）部署支持 Outlines 的 vLLM 服务
```python
 python -m vllm.entrypoints.openai.api_server --model "Meta-Llama-3.1-8B-Instruct" --trust-remote-code --dtype float16 --tensor-parallel-size 2  --max-model-len 8096   --guided-decoding-backend=outlines
```

安装依赖： pip install openai     https://github.com/openai/openai-python/tree/main
```python
prompt = """# 内容 
{content}

# 任务 
判断上述内容的情感。

extra_body = {
    "guided_choice": ['其他', '喜好', '悲伤', '厌恶', '愤怒', '高兴']
}

from openai import OpenAI
import os
client = OpenAI(
    api_key= "EMPTY",
    base_url= "http://192.168.0.172:8000/v1"
)
print(client)  # list models

llm_response = client.chat.completions.create(
            messages=[ {"role": "user", "content": prompt.format(content="今天天气很好")}],
            model="Meta-Llama-3.1-8B-Instruct",
            extra_body=extra_body )

response_str = llm_response.choices[0].message.content
print(response_str)
```


五） 推理模型的输出将推理部分文本格式化输出到 reasoning_content字段， 符合 openai格式
**--reasoning-parser 是 vLLM 中的一个参数，用于指定推理解析器（reasoning parser），其作用是解析模型生成的推理内容（reasoning_content）并将其格式化
该参数需与 --enable-reasoning 一同使用**，后者用于启用模型的推理能力（即支持返回 reasoning_content 字段）。


✅ 使用场景
当你部署的是带有逐步推理能力的模型（如 DeepSeek-R1），并希望 API 返回结构化的推理字段（如 reasoning_content），就需要指定 --reasoning-parser 来告诉 vLLM 如何解析这些输出。

```python
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /new_data/hhs/model_hub/baichuan-inc/Baichuan-M2-32B-GPTQ-Int4 \
  --served-model-name Baichuan-M2-32B-GPTQ-Int4 \
  --tensor-parallel-size 2 \
  --port 8102 \
  --max-model-len 7000 \
  --gpu-memory-utilization 0.8 \
  --uvicorn-log-level info \
  --reasoning-parser qwen3
  
```
