项目地址: https://github.com/ggml-org/llama.cpp/releases#release-b10881

## 安装
1. 确定环境信息
```
# 查询显卡驱动版本，和显卡类型
root@maizi:~# nvidia-smi 查询
Thu Sep 10 13:30:35 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.20             Driver Version: 580.126.20     CUDA Version: 13.0     |
|   0  Tesla T4                       Off |   00000000:5E:00.0 Off |                    0 |
| N/A   69C    P0             63W /   70W |    4377MiB /  15360MiB |     98%      Default |

# 查询CUDA版本
root@maizi:/home# nvcc --version
Cuda compilation tools, release 11.7, V11.7.64
```

2. 升级cuda ，下载匹配的 llama.cpp  版本
```
# 1) 安装 CUDA 12.4 Toolkit（不影响现有 11.7）
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run --toolkit --silent

# 2) 配置环境变量（优先使用 12.4）
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' | sudo tee /etc/profile.d/cuda-12.4.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda-12.4.sh
source /etc/profile.d/cuda-12.4.sh

# 3) 验证
**nvcc --version   # 应该显示 12.4**
**root@maizi:/home# nvcc --version**
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0


# 4) 下载 llama.cpp b10881 CUDA 12.4 预编译包
cd /home
wget -O llama-b10881-bin-ubuntu-x64.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b10881/llama-b10881-bin-ubuntu-x64.tar.gz

tar -xzf llama-b10881-bin-ubuntu-x64.tar.gz -C /opt
sudo ln -sf /opt/llama-b10881/bin/* /usr/local/bin/

llama-cli --version
llama-server --version
```

**检查GPU情况**
```
只要出现 `found 1 CUDA devices` 和 `offloaded 65/65 layers to GPU` 才说明GPU被使用了

docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  ghcr.io/ggml-org/llama.cpp:server-cuda   # 查看容器里是否可以看到显卡
nvidia-smi                                  # 驱动是否正常
docker info | grep -i runtime               # 有没有 nvidia runtime
dpkg -l | grep nvidia-container-toolkit     # 包在不在
```
## 部署说明
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
docker run -d \
  --name llama-server \
  -p 8080:8080 \
  -v /home/qyc/bert/Qwen3.8-27B-Uncensored-IQ4-XS-MTP-16GB-VRAM-GGUF:/models \
  --gpus all \
  ghcr.io/ggml-org/llama.cpp:server-cuda12 \
  -m /models/Qwen3.8-27B-Uncensored-IQ4_XS_4BPW.gguf \
  -c 8192 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
  
  ghcr.io/ggml-org/llama.cpp:server-cuda12
  
  
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