## 代码仓库

https://github.com/ModelTC/lightllm

---

## 支持模型列表

| 模型 | 备注 |
|------|------|
| BLOOM | |
| LLaMA | |
| LLaMA V2 | |
| StarCoder | |
| Qwen-7b | 需要设置参数 `--eos_id 151643 --trust_remote_code` |
| ChatGLM2-6b | 需要设置参数 `--trust_remote_code` |
| Baichuan2-7b | 需要设置参数 `--trust_remote_code` |
| Baichuan2-13b | |
| Baichuan-13b | |
| InternLM-7b | 需要设置参数 `--trust_remote_code` |
| Yi-34b | |

---

## Docker 容器启动

```bash
docker run -it --name lightllm --gpus all -p 8113:8113 --shm-size 1g -v /home/qyc/bert:/data/ ghcr.io/modeltc/lightllm:main /bin/bash
```

## 启动服务

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m lightllm.server.api_server --model_dir /data/chatglm3-6b --host 0.0.0.0 --port 8113 --tp 2 --max_total_token_num 120000 --trust_remote_code
```

<details>
<summary>启动日志</summary>

```
Model kv cache using mode normal
INFO:     Started server process [631]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

</details>

---

## 请求 LLM 服务

```bash
curl http://127.0.0.1:8113/generate -X POST -d '{"inputs":"你好","parameters":{"max_new_tokens": 128}}' -H 'Content-Type: application/json'
```

---

## 参数说明

`lightllm.server.api_server` 支持参数介绍：

| 参数                                   | 说明                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------- |
| `--host`                             | 服务监听地址                                                                     |
| `--port`                             | 服务监听端口                                                                     |
| `--model_dir`                        | 模型权重目录路径，应用将从此目录加载配置、权重和 tokenizer                                         |
| `--tokenizer_mode`                   | tokenizer 加载模式，可选 `slow` 或 `auto`。slow 模式加载快但运行慢，适合调试和测试；追求最佳性能时使用 auto 模式 |
| `--load_way`                         | 模型权重加载方式，默认为 HF（Huggingface 格式），LLaMA 还支持 DS（Deepspeed）                    |
| `--max_total_token_num`              | GPU 和模型可支持的最大 token 数，等于 `max_batch * (input_len + output_len)`            |
| `--batch_max_tokens`                 | 新批次的最大 token 数，控制 prefill 批次大小以防止 OOM                                      |
| `--eos_id`                           | 结束符 stop token id                                                          |
| `--running_max_req_size`             | 同一时间转发请求的最大数量                                                              |
| `--tp`                               | 模型张量并行大小，默认为 1                                                             |
| `--max_req_input_len`                | 请求输入 token 的最大长度                                                           |
| `--max_req_total_len`                | `req_input_len + req_output_len` 的最大值                                      |
| `--nccl_port`                        | 用于构建 PyTorch 分布式环境的 nccl_port                                              |
| `--mode`                             | 模型模式，见下方详细说明                                                               |
| `--trust_remote_code`                | 是否允许加载 Hub 上自定义模型代码                                                        |
| `--disable_log_stats`                | 禁用吞吐量日志统计                                                                  |
| `--log_stats_interval`               | 日志统计间隔（秒）                                                                  |
| `--router_token_ratio`               | 控制路由调度的 token 比率                                                           |
| `--router_max_new_token_len`         | 路由器请求的最大新 token 长度                                                         |
| `--no_skipping_special_tokens`       | 解码时是否跳过特殊 token                                                            |
| `--no_spaces_between_special_tokens` | 解码时是否在特殊 token 之间添加空格                                                      |



## mode 模式说明
[triton_int8kv | ppl_int8kv | triton_flashdecoding]
[triton_int8weight | triton_int4weight | lmdeploy_int4weight | ppl_int4weight]

- **triton_flashdecoding**：适用于长上下文，目前支持 llama、llama2、qwen
- **triton_int8kv**：使用 int8 存储 kv cache，增加 token 容量，使用 triton kernel
- **ppl_int8kv**：使用 int8 存储 kv cache，使用 ppl fast kernel
- **triton_int8weight / triton_int4weight / lmdeploy_int4weight / ppl_int4weight**：使用 int8 或 int4 存储权重