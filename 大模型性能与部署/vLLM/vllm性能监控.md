## 一）vllm自带并行测试工具
`vllm bench serve`是 vLLM 提供的在线服务吞吐量基准测试工具，用于测试已运行的 vLLM 服务器性能。
##### 使用自定义数据集的方法
自定义数据集需要保存为 **JSONL 格式**（每行一个 JSON 对象），必须包含 `prompt` 字段
基本格式示例 (`data.jsonl`)：
```jsonl
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```
**先启动一个vllm服务**
```bash
# 运行自定义数据集基准测试
vllm bench serve \
    --dataset-name custom \
    --dataset-path /path/to/your/data.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --custom-output-len 256 \
    --num-prompts 100 \
    --max-concurrency 32
```
##### 常用参数说明
1. 基础配置参数（Backend & Server）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--backend` | str | `vllm` | 后端服务类型。可选：`vllm`, `openai`, `openai-chat`, `openai-audio`, `openai-embeddings`, `openai-embeddings-chat`, `openai-embeddings-clip`, `openai-embeddings-vlm2vec`, `infinity-embeddings`, `infinity-embeddings-clip`, `vllm-rerank`  |
| `--base-url` | str | `None` | 服务器或 API 的基础 URL（用于外部 API，如 OpenAI），例如 `http://host:port`  |
| `--host` | str | `127.0.0.1` | 服务器主机地址。本地测试推荐用 `127.0.0.1` 强制 IPv4，避免 `localhost` 解析为 IPv6  |
| `--port` | int | `8000` | 服务器端口，默认为 vLLM 的 8000  |
| `--endpoint` | str | `/v1/completions` | API 端点路径，如 `/v1/chat/completions` 或 `/v1/completions`  |
| `--model` | str | `None` | 模型名称。如果未指定，将从服务器的 `/v1/models` 端点获取第一个模型  |
| `--served-model-name` | str | `None` | API 中使用的模型名称。如果未指定，则模型名称将与 `--model` 参数相同  |
| `--tokenizer` | str | `None` | 分词器的名称或路径，如果未使用默认分词器  |
| `--tokenizer-mode` | str | `auto` | 分词器模式。可选：`auto`, `hf`, `slow`, `mistral`, `deepseek_v32` 或其他自定义值  |
| `--trust-remote-code` | bool | `False` | 信任来自 Huggingface 的远程代码  |
| `--lora-modules` | str | `None` | 启动服务器时传入的 LoRA 模块名称子集。对于每个请求，脚本都会随机选择一个 LoRA 模块  |

2. 测试负载参数（Load Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-prompts` | int | `1000` | 要处理的提示（请求）数量  |
| `--max-concurrency` | int | `None` | 最大并发请求数。用于模拟高级别组件强制执行最大并发请求数的环境。虽然 `--request-rate` 控制请求的启动速率，但此参数控制一次实际允许执行多少请求  |
| `--request-rate` | float | `inf` | 每秒请求数（RPS）。如果为 `inf`，所有请求在时间 0 发送。否则使用泊松过程或伽马分布合成请求到达时间  |
| `--burstiness` | float | `1.0` | 请求生成的突发性因子。仅在 `request_rate` 不是 `inf` 时生效。`1.0` 遵循泊松过程，`0 < burstiness < 1` 更突发，`burstiness > 1` 更均匀  |
| `--seed` | int | `0` | 随机种子，用于可复现的测试结果  |

3. 数据集选择参数（Dataset Selection）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset-name` | str | `random` | 要进行基准测试的数据集名称。可选：`sharegpt`, `burstgpt`, `sonnet`, `random`, `random-mm`, `random-rerank`, `hf`, `custom`, `custom_mm`, `prefix_repetition`, `spec_bench`  |
| `--dataset-path` | str | `None` | sharegpt/sonnet 数据集的路径。如果使用 HF 数据集，则为 Huggingface 数据集 ID  |
| `--no-stream` | bool | `False` | 不要以流式模式加载数据集  |
| `--no-oversample` | bool | `False` | 如果数据集样本少于 `num-prompts`，则不进行过采样  |
| `--skip-chat-template` | bool | `False` | 跳过将聊天模板应用于支持它的数据集的提示  |
| `--enable-multimodal-chat` | bool | `False` | 为支持多模态聊天的相关数据集启用多模态聊天转换  |
| `--disable-shuffle` | bool | `False` | 禁用数据集样本的洗牌以实现确定性排序  |
| `--input-len` | int | `None` | 数据集的通用输入长度。映射到特定数据集的输入长度参数（如 `--random-input-len`）。如果未指定，使用数据集默认值  |
| `--output-len` | int | `None` | 数据集的通用输出长度。映射到特定数据集的输出长度参数（如 `--random-output-len`）。如果未指定，使用数据集默认值  |
| `--random-input-len` | int | `1024` | 随机数据集的输入 token 数  |
| `--random-output-len` | int | `1024` | 随机数据集的输出 token 数  |
| `--custom-output-len` | int | `256` | 每个请求的输出 token 数。除非设置为 `-1`，否则覆盖从数据集中加载的潜在输出长度  |
| `--custom-skip-chat-template` | bool | `False` | 跳过应用 chat template（如果数据已包含模板） |


4. 请求速率预热参数（Ramp-up）
用于压力测试或寻找最大吞吐量 ：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ramp-up-strategy` | str | `None` | 预热策略。可选：`linear`（线性增加）, `exponential`（指数增加） |
| `--ramp-up-start-rps` | float | `None` | 预热的起始请求速率（RPS） |
| `--ramp-up-end-rps` | float | `None` | 预热的结束请求速率（RPS） |

5. 结果输出与指标参数（Results & Metrics）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ignore-eos` | bool | `False` | 在发送基准测试请求时设置 ignore_eos 标志。警告：deepspeed_mii 和 tgi 不支持  |
| `--percentile-metrics` | str | `ttft,tpot,itl` | 用于报告百分位数的选定指标的逗号分隔列表。可选：`ttft`, `tpot`, `itl`, `e2el`  |
| `--metric-percentiles` | str | `99` | 选定指标的百分位数的逗号分隔列表。例如 `25,50,75` 报告 P25, P50, P75  |
| `--goodput` | str | `None` | 为 goodput 指定服务水平目标，格式为 `KEY:VALUE` 对（如 `ttft:100` 表示 TTFT < 100ms）。多个对用空格分隔  |
| `--label` | str | `None` | 基准测试结果的标签（前缀）。如果未指定，使用 `--backend` 的值  |
| `--request-id-prefix` | str | `bench-xxxxx-` | 请求 ID 的前缀  |

6. 结果保存参数（Result Saving）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save-result` | bool | `False` | 将基准测试结果保存到 JSON 文件  |
| `--save-detailed` | bool | `False` | 保存结果时，是否包含每个请求的信息（响应、错误、ttfts、tpots 等） |
| `--append-result` | bool | `False` | 将基准测试结果追加到现有的 JSON 文件  |
| `--result-dir` | str | `None` | 保存基准测试 JSON 结果的目录。如果未指定，保存在当前目录  |
| `--result-filename` | str | `None` | 保存基准测试 JSON 结果的文件名。如果未指定，使用 `{label}-{args.request_rate}qps-{base_model_id}-{current_dt}.json` 格式  |
| `--metadata` | str | `None` | 键值对（如 `--metadata version=0.3.3 tp=1`）用于此运行的元数据，将保存在结果 JSON 文件中  |

7. 其他高级参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-beam-search` | bool | `False` | 使用束搜索  |
| `--logprobs` | int | `None` | 要计算并返回的 logprobs-per-token 数量  |
| `--num-warmups` | int | `0` | 预热请求的数量  |
| `--profile` | bool | `False` | 使用 vLLM 剖析。必须在服务器上提供 `--profiler-config`  |
| `--ready-check-timeout-sec` | int | `0` | 等待端点就绪的最长时间（秒）。默认跳过就绪检查  |
| `--extra-body` | str | `None` | 要在每个请求中包含的额外正文参数的 JSON 字符串。例如 `'{"chat_template_kwargs":{"enable_thinking":false}}'`  |
| `--disable-tqdm` | bool | `False` | 禁用 tqdm 进度条  |
| `--header` | str | `None` | 随每个请求传递的头部，格式为键值对（如 `--header x-additional-info=0.3.3`）。会覆盖后端常量和环境变量设置的值  |

##### 输出指标说明
运行后会输出以下关键性能指标：
- **Request throughput (req/s)**: 请求吞吐量
- **Input token throughput (tok/s)**: 输入 token 吞吐量
- **Output token throughput (tok/s)**: 输出 token 吞吐量
- **Mean TTFT (ms)**: 首 token 延迟平均值（Time To First Token）
- **Mean TPOT (ms)**: 每输出 token 延迟平均值（Time Per Output Token）
- **Mean ITL (ms)**: token 间延迟平均值（Inter-token Latency）


## 资源监控
##### vLLM 服务指标（通过 `/metrics` 端点 http://localhost:8000/metrics ）
**延迟指标**：
- **TTFT** (Time To First Token): 首 token 时间 = 排队时间 + 预填充时间
- **ITL** (Inter-Token Latency): token 间延迟，反映解码阶段性能
- **E2E Latency**: 端到端总延迟
**吞吐量指标**：
- `vllm:prompt_tokens_total`: 输入 token 总数
- `vllm:generation_tokens_total`: 生成 token 总数
- Token 吞吐量 (tokens/s)
**服务健康指标**：
- `vllm:num_requests_running`: 正在运行的请求数（当前 batch size）
- `vllm:num_requests_waiting`: 等待队列深度
- `vllm:num_preemptions_total`: 请求被抢占次数（内存压力指标）

##### python脚本监控资源情况
![[未命名.base]]
