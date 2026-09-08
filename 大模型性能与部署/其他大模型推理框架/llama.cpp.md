vLLM **原生不支持 GGUF 格式**。vLLM 的生态建立在 **HuggingFace safetensors** 格式上，主要支持以下量化方案：

|格式|vLLM 支持？|
|---|---|
|Safetensors (原生)|✅ 完全支持|
|AWQ|✅ 支持|
|GPTQ|✅ 支持|
|FP8 (原生HF)|✅ 支持|
|**GGUF** (llama.cpp)|**❌ 不支持**|
|IQ4_XS (GGUF特有)|**❌ 不支持**|

GGUF 格式是 **llama.cpp 生态** 的专有格式，IQ4_XS 也是 llama.cpp 独有的量化方法。

```
docker run -p 8080:8080 \
  -v /home/qyc/bert/Qwen3.8-27B-Uncensored-IQ4-XS-MTP-16GB-VRAM-GGUF:/models \
  --gpus all \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-Uncensored-IQ4_XS_4BPW.gguf \
  -c 10000 \
  --host 0.0.0.0 \
  --port 8080 
  
curl http://localhost:8080/v1/models

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3.8-27B-Uncensored-IQ4_XS_4BPW.gguf",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```
  


```
docker run -p 8080:8080 \
  -v /home/qyc/bert/Qwen3.8-27B-GGUF:/models \
  --gpus all \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-UD-IQ1_M.gguf \
  -ngl 99 \
  -c 28192 \
  -b 512 \
  -ub 512 \
  -t 4 \
  -tb 4 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning off \
  --chat-template-kwargs '{"reasoning_effort": "low"}' \
  --temp .6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --host 0.0.0.0 \
  --port 8080
```