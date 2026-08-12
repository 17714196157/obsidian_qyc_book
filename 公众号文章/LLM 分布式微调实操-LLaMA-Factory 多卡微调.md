---
title: "LLM 分布式微调实操-LLaMA-Factory 多卡微调"
source: "https://mp.weixin.qq.com/s/cjbvVPalrJoLMJoc-T-NcA"
author:
  - "[[小星小浩]]"
published:
created: 2026-08-12
description: "上一篇我们搞清楚了为什么需要分布式训练、四种并行方式的区别，以及 DeepSpeed 框架 + ZeRO 怎么解决显存问题。本文带你 LLM 分布式训练实操-LLaMA-Factory 多卡训练实战"
tags:
  - "clippings"
---
原创 小星小浩 *2026年7月7日 08:29*

上一篇我们搞清楚了 **为什么需要分布式训练** 、 **四种并行方式的区别** ，以及 **DeepSpeed 框架 + ZeRO** 怎么解决显存问题。

本篇进入实操。 **LLaMA-Factory** 在两卡环境上完成 Qwen2.5-1.5B-Instruct 的多卡微调。

需要本章 **配套源码和数据集** 的同学，可以 **点❤️ + 关注** ，我会把完整工程发给你。

上一篇快速导航：《 [（十三）大模型分布式训练（上）：分布式训练基本概念 + DeepSpeed 框架](https://mp.weixin.qq.com/s?__biz=MzYzNjI2NjMyNA==&mid=2247485721&idx=1&sn=ae6c962785276eeca0a4ba5a8aca1389&scene=21#wechat_redirect) 》

---

## 1\. 准备工作

### 1.1 确认 LLaMA-Factory 已安装

按第八篇的流程，LLaMA-Factory 已安装：

```
conda activate llama_factory
cd /root/autodl-tmp/LLaMA-Factory
```

本章路径 `/root/autodl-tmp/` 为例，本地环境替换为自己的目录即可。

### 1.2 确认多卡环境

首先用 `nvidia-smi` 或更友好的 `nvitop` 看下当前机器的 GPU 情况：

```
# 查看 GPU 数量、显存占用
nvidia-smi

# 或用 nvitop
nvitop
```

**看到几张卡，就说明最多能跑几卡并行** （实际可用数受显存限制）。
[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/213924da95a828a2209b790806b4ffb4_MD5.png|Open: file-20260812104351121.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/213924da95a828a2209b790806b4ffb4_MD5.png]]

nvitop 查看 GPU 状态

**我的设备配置信息**

[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/5dbbfb4e45523324ec4d431f6bb31c4b_MD5.png|Open: file-20260812104404321.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/5dbbfb4e45523324ec4d431f6bb31c4b_MD5.png]]

### 1.3 安装 DeepSpeed

LLaMA-Factory 本身依赖 `deepspeed` 和 `accelerate` 两个包：

```
pip install deepspeed accelerate
```

---

## 2\. 启动 WebUI

```
cd /root/autodl-tmp/LLaMA-Factory
llamafactory-cli webui
```

启动后访问 `http://localhost:7860/` ，进入训练配置页。

[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/0434be066a5a3c8239e50cd3cb30929c_MD5.png|Open: file-20260812104415403.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/0434be066a5a3c8239e50cd3cb30929c_MD5.png]]

---

## 3\. 多卡训练关键配置

**重要前提：**

LLaMA-Factory 支持 **三种分布式训练引擎** —— **NativeDDP** （PyTorch 原生 DDP）、 **DeepSpeed** 、 **FSDP / FSDP2** （PyTorch 全切片并行）。

本文聚焦最常用的 **DeepSpeed** 。

**三种引擎的能力对比：**

| 引擎               | 数据切分 | 模型切分 | 优化器切分 | 参数卸载 | 推荐场景                        |
| ---------------- | ---- | ---- | ----- | ---- | --------------------------- |
| **NativeDDP**    | ✅    | ❌    | ❌     | ❌    | 模型能装进单卡，纯数据并行加速             |
| **DeepSpeed**    | ✅    | ✅    | ✅     | ✅    | 显存紧张、需要 ZeRO 切分（ **本文主线** ） |
| **FSDP / FSDP2** | ✅    | ✅    | ✅     | ✅    | PyTorch 原生方案，70B+ 大模型备选     |

### 3.1 WebUI 上的"分布式训练参数"

在 WebUI 的 **Train** 标签页，往下拉会看到 **分布式训练参数** （DeepSpeed 相关）：

| 配置项                      | 填写内容               | 说明                                |
| ------------------------ | ------------------ | --------------------------------- |
| **DeepSpeed stage**      | `zero2`  或 `zero3` | 推荐 zero2，速度快；显存实在不够再选 zero3       |
| **使用 DeepSpeed Offload** | 按需勾选               | 会显著降低显存，但 **训练速度会变慢** （数据搬运到 CPU） |
| **设备数量**                 | 自动识别，无需手填          | LLaMA-Factory 会自动用上所有可见 GPU       |

**WebUI 多卡配置位置：**
[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/222835f1fecaa03b35fcf655d5df4f1b_MD5.png|Open: file-20260812104446691.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/222835f1fecaa03b35fcf655d5df4f1b_MD5.png]]

LLaMA-Factory DeepSpeed 配置

**关键理解：** LLaMA-Factory 的"DeepSpeed stage"下拉框本质就是选了不同的 ZeRO 切分粒度。下游会自动生成对应配置文件，无需手写。

### 3.2 怎么选 stage？（速查表）

| 显存压力 | 推荐 stage | 附加建议 |
| --- | --- | --- |
| 单卡 24GB 跑 7B | ZeRO-2 + BF16 | 不需要 offload |
| 单机 4 卡 24GB 跑 13B | ZeRO-3 | 开 offload 会更稳，速度更快 |
| 单机 8 卡 80GB 跑 70B | ZeRO-3 + CPU offload | 必然要 |
| **微调 1.5B ~ 7B** | **DDP（不开 ZeRO）即可** | **ZeRO-2 也行** |

本章案例用的是 Qwen2.5-1.5B-Instruct（1.5B），单卡就能装下。多卡的目的主要是"加速训练"——选 `zero2` 即可。

### 3.3 典型配置推荐

| 场景 | 模型 | 显卡 | DeepSpeed stage | Offload | 批大小 |
| --- | --- | --- | --- | --- | --- |
| 单卡够用 | 1.5B | 24GB | 不开 | — | 2 |
| 多卡加速 | 1.5B | 2 × 24GB | zero2 | 不开 | 2 × 2 卡 |
| 多卡训练 7B | 7B | 4 × 24GB | zero3 | 可选 | 1 × 4 卡 |
| 多机训练 13B | 13B | 8 × 80GB | zero3 | 建议开 | 1 × 8 卡 |

上面所有配置都基于 **DeepSpeed** 。如果你的场景超出本文范围（30B+ 模型、消费级显卡大模型微调等），可以看文章末尾的 **9\. 进阶方案** 节，那里讲了 FSDP 和 DeepSpeed AutoTP 两个扩展方向。

---

## 4\. 训练参数配置（沿用第十一篇 identity + fintech 案例）

为保持连续性， **数据集、超参数沿用第十一篇的双数据集微调配置** ：

| 配置项 | 填写内容 |
| --- | --- |
| **模型名称** | `Qwen2.5-1.5B-Instruct` |
| **模型路径** | `/root/autodl-tmp/models/Qwen/Qwen2.5-1.5B-Instruct` |
| **微调方法** | LoRA |
| **数据集** | `identity_my,fintech` |
| **对话模板** | `qwen` |
| **截断长度** | `1024` |
| **批处理大小** | `2 （每卡）` |
| **学习率** | `2e-5` |
| **Epochs** | `10000` |
| **计算类型** | `bf16` |
| **验证集比例** | `0.05` |
| **DeepSpeed stage** | `zero2 （多卡推荐起点）` |

  
[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/94a1006bfe09f194fce8e80860afda9c_MD5.png|Open: file-20260812104515178.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/94a1006bfe09f194fce8e80860afda9c_MD5.png]]

Train配置参数

**注意：**

多卡训练时 `per_device_train_batch_size` 是 **每张卡** 的 batch size，全局 batch = `per_device × 卡数 × gradient_accumulation` 。比如 2 卡 + batch=2 + 累积=8 → 全局 batch = 32。

---

## 5\. 启动分布式训练

### 5.1 点击 Start

点 **Start** 后，LLaMA-Factory 会自动生成等效命令（点 **Preview Command** 可看）：

```
# 2 卡 + ZeRO-2 的等效命令（只展示关键参数；完整命令以 WebUI Preview Command 生成的为准）
llamafactory-cli train \
    --model_name_or_path /root/autodl-tmp/model/Qwen/Qwen2___5-1___5B-Instruct \
    --finetuning_type lora \
    --template qwen \
    --dataset identity_my,fintech \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 2000.0 \
    --max_samples 100000 \
    --save_steps 100 \
    --output_dir saves/Qwen2.5-1.5B-Instruct/lora/train_2026-07-01-20-20-49 \
    --quantization_bit 8 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --val_size 0.05 \
    --eval_steps 100 \
    --deepspeed llamaboard_cache/ds_z2_config.json
    ...
```

执行以上的命令，是一样的效果。

除了 `llamafactory-cli` 指令方式，还有 `torchrun` 、 `accelerate` 指令方式。

**torchrun 命令**

使用 torchrun 指令启动 NativeDDP 引擎进行单机多卡训练。下面提供一个示例（只显示了关键参数）：

```
torchrun  --standalone --nnodes=1 --nproc-per-node=8  src/train.py \
--stage sft \
--model_name_or_path /root/autodl-tmp/model/Qwen/Qwen2___5-1___5B-Instruct \
--do_train \
--dataset identity_my,fintech \
--template qwen \
--finetuning_type lora \
--output_dir  saves/Qwen2.5-1.5B-Instruct/lora/train_2026-07-01-21-30-25 \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 5 \
--save_steps 100 \
--learning_rate 5e-05 \
--num_train_epochs 2000.0 \
--plot_loss True \
--bf16
...
```

**accelerate 命令**

这里就不展开讲了，详情参考官方文档。

### 5.2 验证多卡已启动

训练启动后， **Output** 标签页会显示日志。 **日志中没有直接写出"2 块 GPU"这行字** ，但通过以下三行日志的 **数学关系** 可以准确反推出实际用了几张卡：
[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/74c6d4e16a7ab009bb2725443fd7b3d2_MD5.png|Open: file-20260812104557240.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/74c6d4e16a7ab009bb2725443fd7b3d2_MD5.png]]
WebUI 训练日志

- **Instantaneous batch size per device = 2** （每张卡每次处理 2 条）
- **Gradient Accumulation steps = 8** （梯度累积 8 步）
- **Total train batch size... = 32** （总等效批次 32）

**计算公式：**

```
GPU 数量 = 总批次 / (每卡批次 × 累积步数) = 32 / (2 × 8) = 2
```

反推出来 = **2 张卡** ，与你 WebUI 里设置的"多卡加速"完全一致 → 多卡训练已正常启动。卡数为 4 / 8 时同理反推即可。

---

## 6\. 训练监控

**多卡训练时，每张卡都应该均匀工作** 。可以用 `nvidia-smi` 实时观察：
[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/80bafbc0d6e4e2f69b689da760d42072_MD5.png|Open: file-20260812104541542.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/80bafbc0d6e4e2f69b689da760d42072_MD5.png]]

训练中 GPU 状态

**正常情况：**

- 所有卡的 `GPU-Util` **接近一致**
- 显存占用接近平均分配
- 日志中 step 耗时稳定

**异常情况：**

- 只有部分卡 GPU-Util 高 → 数据/模型没均匀切分
- 显存差异大 → 可能是 ZeRO 没生效

---

## 7\. Checkpoint 保存路径

多卡训练完成后，checkpoint 路径与单卡 **完全一致** ：

```
saves/Qwen2.5-1.5B-Instruct/lora/train_2026-07-01-XX-XX-XX/
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
└── adapter_model.safetensors   ← 最终 LoRA 权重
```

**易踩坑：** 训练时多卡，但推理/合并时如果想用单卡，记得把 `--deepspeed` 去掉（合并阶段通常不需要分布式）。

---

## 8\. 指定特定卡训练（进阶）

如果机器上有 8 张卡，但只想用其中 2 张（比如 0 和 2），有两种方式：

### 8.1 WebUI 方式

WebUI 默认用所有可见卡， **目前 LLaMA-Factory WebUI 不直接支持指定卡** ，需要走命令行。

### 8.2 命令行 + CUDA\_VISIBLE\_DEVICES

```
# 指定使用 GPU 0 和 GPU 2
# 注意：多卡训练必须加 FORCE_TORCHRUN=1，强制 llamafactory-cli 调用 torchrun 启动多进程
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,2 llamafactory-cli train \
    examples/train_lora/qwen2.5_1.5b_lora_sft.yaml
```

**我测试的示例**

```
# 我测试的示例
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/train_lora/qwen2.5_1.5b_lora_sft.yaml
```

**qwen2.5\_1.5b\_lora\_sft.yaml 部分内容**

```
### model
model_name_or_path:/root/autodl-tmp/model/Qwen/Qwen2___5-1___5B-Instruct
trust_remote_code:true

### method
stage:sft
do_train:true
finetuning_type:lora
lora_rank:8

### dataset
dataset:identity_my,fintech
template:qwen
cutoff_len:1024
max_samples:2000

### output
output_dir:saves/Qwen2.5-1.5B-Instruct/lora/train_2026-07-01-21-30-49
logging_steps:5
save_steps:100

### train
per_device_train_batch_size:2
gradient_accumulation_steps:8
learning_rate:5e-05
num_train_epochs:2000

### eval
val_size:0.05
```
  
[[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/feb1a4977e89fbab2beed9e03ca3321e_MD5.png|Open: file-20260812104624264.png]]
![[公众号文章/assets/LLM 分布式微调实操-LLaMA-Factory 多卡微调/feb1a4977e89fbab2beed9e03ca3321e_MD5.png]]

训练日志

`1 distributed tasks` 表示启动了1 个分布式进程（即单进程单卡模式）。

`device: cuda:0` 使用 GPU 0。

`world size: 1` 总进程数为 1，说明没有启动多卡分布式（多卡时 world size 会大于 1）。

**关键提示：**

LLaMA-Factory 用 `llamafactory-cli train` 启动多卡时，\*\*必须设置 `FORCE_TORCHRUN=1` \*\*，否则不会自动调用 torchrun 启动多进程——这是官方文档明确要求的。

**YAML 文件名核对：**

LLaMA-Factory `examples/train_lora/` 下的 YAML 命名随版本变化，请按你本地仓库实际文件名填写。

常见命名如 `qwen2_5_lora_sft.yaml` / `qwen3_lora_sft.yaml` / `llama3_lora_sft.yaml` 。

如果找不到现成的，可以复制官方示例后修改模型路径、数据集名等关键字段。

### 8.3 YAML 方式 + deepspeed 配置

```
# examples/train_lora/qwen2.5_1.5b_lora_sft.yaml 末尾追加
deepspeed: examples/deepspeed/ds_z2_config.json
```

然后：

```
# YAML 方式同样需要 FORCE_TORCHRUN=1
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,2 llamafactory-cli train \
    examples/train_lora/qwen2.5_1.5b_lora_sft.yaml
```

**提示：**

训练时 DeepSpeed 会自动检测可见卡数，无需在 YAML 里写死。

但如果用 FSDP 引擎， `fsdp_config.yaml` 里的 `num_processes` 必须等于实际总卡数，否则会启动失败。

### 8.4 多机多卡方式

多机多卡方式，同样有 `llamafactory-cli` 、 `torchrun` 、 `accelerate` 这几种指令方式。

详见官方文档：https://llamafactory.readthedocs.io/zh-cn/latest/advanced/distributed.html

---

## 9\. 进阶方案：FSDP 与 DeepSpeed AutoTP（延伸阅读）

**本节是延伸阅读** ，对应 3.x 节开头的引擎对比表中除 DeepSpeed 之外的两种方案。1.5B 训练用不到，但模型规模到 7B+ 时可以关注。

### 9.1 备选方案：FSDP（FSDP / FSDP2）

如果机器装了较新版本的 PyTorch（≥ 2.4），可以用 **FSDP** （Fully Sharded Data Parallel）——PyTorch 原生的全切片并行方案， **不需要额外装 DeepSpeed** 。

FSDP 的关键参数是 **ShardingStrategy** ：

| ShardingStrategy | 切分内容 | 对应 DeepSpeed 阶段 |
| --- | --- | --- |
| `FULL_SHARD` | 参数 + 梯度 + 优化器状态全切 | 类似 ZeRO-3 |
| `SHARD_GRAD_OP` | 仅切梯度和优化器状态，参数不切 | 类似 ZeRO-2 |
| `NO_SHARD` | 全部不切 | 类似 ZeRO-0 |

**FSDP + QLoRA** 是消费级显卡跑 70B+ 模型的常用组合（LLaMA-Factory 提供了 `examples/extras/fsdp_qlora/` 示例）。 **⚠️ 警告：FSDP+QLoRA 不能搭配 GPTQ/AWQ 量化模型** 。

**FSDP2** （最新版本，基于 DTensor）通信与计算重叠效率更高，启动方式与 FSDP 类似，配置上多一行 `fsdp_version: 2` 。

### 9.2 新动向：DeepSpeed AutoTP（DeepSpeed ≥ 0.16.4）

DeepSpeed 0.16.4+ 新增了 **AutoTP** （自动张量并行），可与 ZeRO-1 或 ZeRO-2 组合使用：

- 通信开销比 ZeRO-3 低，显存节省效果接近 ZeRO-3
- 支持更大 batch size 和更长上下文
- 配置文件示例： `examples/deepspeed/ds2_autotp.json`
```
{
    "ZeRO_optimization": {
        "stage": 2
    },
    "tensor_parallel": {
        "autotp_size": 4
    }
}
```

目前模型支持受限，需查 AutoTP 支持列表(

https://www.deepspeed.ai/tutorials/automatic-tensor-parallelism/#supported-models)。

**1.5B 训练用不到，但跑 30B+ 模型时可以关注这个方向** 。  

---

## 10\. 常见问题

### 10.1 启动报"no distributed launcher found"

**原因：** 没装 `deepspeed` 或 `accelerate` 。

```
pip install deepspeed accelerate
```

### 10.2 WORLD\_SIZE 一直是 1

**原因：** 机器实际只有 1 张卡，或 `CUDA_VISIBLE_DEVICES` 没生效。

**排查：**

```
# 确认可见卡数
nvidia-smi -L
```

看到 GPU 0、GPU 1

```
(llama_factory) root@autodl-container:~# nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-c3d2d408-c347-e2d0-3515-8e03eb59d45e)
GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-a65b754f-5c40-9ea1-45da-a3cc80c9dc6c)
```

### 10.3 训练速度比单卡还慢

**原因：** 模型太小（数据并行通信开销 > 计算收益），或 ZeRO-3 通信太频繁。

**解决方案：**

- 模型 < 7B 时 **没必要上 ZeRO-3** ，用 `zero2` 或 `DDP（不开 ZeRO）` 即可
- 把 stage 调到 `zero2` （比 zero3 快）
- 检查 `gradient_accumulation_steps` 是否合理（太小会导致频繁同步）

### 10.4 显存够却 OOM（Out of Memory）

**原因：** 可能是 `cutoff_len` 太大或激活值爆显存。

**排查：**

```
# 降低截断长度
cutoff_len: 1024    # 从 2048 降到 1024

# 开启梯度检查点
gradient_checkpointing: true
```

### 10.5 多机训练时卡住

**原因：** 节点间 NCCL 通信端口没开或防火墙阻断了。

**排查：**

```
# 多机训练前，先确认节点间能互通
ping <master_node_ip>

# 设置端口（默认 29500，需保证节点间都开放）
export MASTER_PORT=29500
```

---

## 11\. 小结

| 阶段 | 关键点 |
| --- | --- |
| 环境准备 | `nvidia-smi`  / `nvitop` 查卡 → `pip install deepspeed accelerate` |
| WebUI 配置 | DeepSpeed stage 选 `zero2` （1.5B~7B 起点） |
| 训练参数 | `per_device_batch × 卡数 × 累积`  \= 全局 batch |
| 启动验证 | 日志首行看 `WORLD_SIZE=N` ，与卡数一致 |
| 训练监控 | 每张卡 GPU-Util 应均匀（80%+） |
| Checkpoint | 多卡与单卡路径一致，推理/合并阶段通常回到单卡 |
| 进阶方案 | 30B+ 可关注 **FSDP + QLoRA** ；DeepSpeed ≥0.16.4 可关注 **AutoTP+ZeRO-1/2** |

**选型速记：** **1.5B-7B LoRA 微调 → 多卡用 ZeRO-2 起步；7B-13B 用 ZeRO-3；70B 必然 ZeRO-3 + Offload** 。

---

## 下一篇预告

业界还有另一个非常流行的微调框架—— **XTuner** （上海 AI Lab 出品）。

**下篇** 将带你 **从零搭建 XTuner 环境** ，完成 Qwen 的 LoRA / QLoRA 微调，并讲透 `pth_to_hf` 和 `merge` 两步转换。

---

**系列导航**

| 篇章 | 主题 |
| --- | --- |
| 上篇 | 分布式训练基本概念 + DeepSpeed 框架 |
| **本篇（中）** | **LLaMA-Factory 多卡训练实操** |
| 下篇 | XTuner 微调全流程 |
