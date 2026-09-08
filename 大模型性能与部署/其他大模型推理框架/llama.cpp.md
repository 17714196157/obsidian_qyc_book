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
### 1）1张T4，启动Q3版本的qwen3.8-27B，大约占13GB
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
  

### 2）2张T4，启动Q4版本的qwen3.8-27B，大约占29GB
```bash
snapshot_download(repo_id='unsloth/Qwen3.8-27B-GGUF',repo_type='model',

                allow_patterns=["Qwen3.8-27B-UD-Q4_K_M.gguf","mmproj-F16.gguf"],  #"Qwen3.8-27B-Q4_K_M.gguf" "Qwen3.8-27B-UD-IQ1_M.gguf",

                  local_dir='/home/qyc/bert/Qwen3.8-27B-GGUF',resume_download=True)
                  
docker run -p 8080:8080 \
  -v /home/Qwen3.8-27B-GGUF:/models \
  --gpus all \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --mmproj /models/mmproj-F16.gguf \
  --alias "Qwen3.8-27B-Q4_K_M" \
  -t 5 \
  -tb 5 \
  -ctk q8_0 \
  -ctv q8_0 \
  -ngl 100 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 8192 \
  --temp 0.8 \
  -b 512 \
  -ub 512 \
  --parallel 1 \
  --flash-attn on \
  -sm tensor \
  --tensor-split 1,1 \
  --image-min-tokens 1024
 
➜  ~ curl http://localhost:8080/v1/models
{"models":[{"name":"Qwen3.8-27B-Q4_K_M","model":"Qwen3.8-27B-Q4_K_M","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion","multimodal"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"Qwen3.8-27B-Q4_K_M","aliases":["Qwen3.8-27B-Q4_K_M"],"tags":[],"object":"model","created":1788910812,"owned_by":"llamacpp","meta":{"vocab_type":true,"n_vocab":248320,"n_ctx":8192,"n_ctx_train":262144,"n_embd":5120,"n_params":27320697856,"size":16453443584,"ftype":"Q4_K - Medium"}}]}#    


```