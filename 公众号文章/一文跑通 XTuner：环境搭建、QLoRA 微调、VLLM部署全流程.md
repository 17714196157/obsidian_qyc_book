---
title: "一文跑通 XTuner：环境搭建、QLoRA 微调、VLLM部署全流程"
source: "https://mp.weixin.qq.com/s?__biz=MzYzNjI2NjMyNA==&mid=2247485795&idx=1&sn=79516648b641c4c3f85ba1a061396eca&chksm=f083bb58c7f4324e325d80c9e02cf291144cd5ec806b89ba88b655c6e85215a5909faec6d6d9&cur_album_id=4506690984139063304&scene=189#wechat_redirect"
author:
  - "[[小星小浩]]"
published:
created: 2026-08-12
description: "本文用 XTuner 走配置驱动路线，一文跑通单卡 QLoRA 全流程，从装环境到 vLLM 上线。附可复现步骤，多卡内容下篇再讲。"
tags:
  - "clippings"
---
原创 小星小浩 *2026年7月17日 08:37*

我们用 LLaMA-Factory 完成了多卡训练。还有另一个非常流行的微调框架—— **XTuner** （上海 AI Lab 出品），走的是 **配置驱动** 路线，适合需要精细控制训练流程的场景。

本篇带你 **从零搭建 XTuner 环境** ，完成 Qwen/Qwen2.5-1.5B-Instruct 的 QLoRA **单卡** 微调，覆盖：

```
主线：环境搭建 → 配置 → 单卡训练 → 监控/续训 → 转换 → 合并 → 推理 → vLLM
```

多卡、指定卡、多机多卡等进阶内容见 **下一篇** 。

需要本章 **配套源码和数据集** 的同学，可以 **点❤️ + 点赞 + 关注** ，我会把完整工程发给你。

上一篇快速导航：《 [LLM 分布式微调实操-LLaMA-Factory 多卡微调](https://mp.weixin.qq.com/s?__biz=MzYzNjI2NjMyNA==&mid=2247485748&idx=1&sn=25772d4675082b2decdc0c30198399f5&scene=21#wechat_redirect) 》

---

## 1\. XTuner 介绍

**![[公众号文章/assets/一文跑通 XTuner：环境搭建、QLoRA 微调、VLLM部署全流程/9b2f84e2adb5ab785e672390a00c1297_MD5.webp]]**

**XTuner** 是上海 AI Lab 出品的 LLM 微调工具箱，通过 `xtuner train` 、 `xtuner list-cfg` 、 `xtuner convert` 等命令完成全流程，特色如下：

| 特色 | 说明 |
| --- | --- |
| **一键安装** | 一键装好 PyTorch、Transformers、DeepSpeed、bitsandbytes 等依赖 |
| **配置驱动** | 所有训练参数写在 Python 配置文件中， **代码即配置** |
| **多机多卡开箱即用** | torchrun / slurm 双模式，与上篇 DeepSpeed 知识完全衔接 |
| **QLoRA 友好** | 内置 4-bit 量化 + LoRA，单卡可微调 7B 模型 |
| **完整转换链** | PTH → HF → Merge 工具齐全 |

**版本说明：** 本篇使用 **0.1.23** ——最后一个自带 `xtuner` 命令行工具的版本。后续上海 AI Lab 推出了架构不同的新一代训练引擎，将在后面章单独讲解。

**与 LLaMA-Factory 的差异：**

| 维度 | LLaMA-Factory | XTuner |
| --- | --- | --- |
| 配置方式 | WebUI / YAML / CLI | Python 配置文件 |
| 学习曲线 | 上手快 | 需读懂 mmengine 配置 |
| 自定义空间 | 中等 | 极高（可直接改模型结构、训练循环） |
| 多卡启动 | 自动 | `NPROC_PER_NODE=N`  显式指定 |
| 适合人群 | 业务应用方 | 算法工程师、研究者 |

想深入研究训练过程选 XTuner。

---

## 2\. 环境准备

确保显卡驱动正确安装即可，例如在 NVIDIA GPU 设备上，nvidia-smi 的 Driver Version 需要大于 550.127.08

XTuner 推荐使用 **conda** 隔离环境，避免依赖冲突。

```
# 创建独立环境（python 3.10 兼容性最好）
conda create --name xtuner-legacy python=3.10 -y
conda activate xtuner-legacy
```

---

## 3\. 安装 XTuner

### 3.1 安装依赖

```
pip install 'xtuner[all]==0.1.23'
```

这一步会安装 XTuner 及其所有依赖（PyTorch、Transformers、DeepSpeed、bitsandbytes 等）， **耗时较长** ，耐心等待。

### 3.2 依赖冲突时的版本修正（可选）

`0.1.23` 会一并安装配套依赖。若训练或导入时报版本不兼容，可手动锁定到稳定组合：

```
pip install 'transformers>=4.36.0,!=4.38.0,!=4.38.1,!=4.38.2,<5.0.0' 'peft>=0.4.0,<0.14.0'
```

### 3.3 验证安装

```
xtuner version
```

正常输出版本号（0.1.23）说明安装成功

```
(xtuner-legacy) root@autodl-container:~/autodl-tmp# xtuner version
07/12 20:59:53 - mmengine - INFO - 0.1.23
```
```
xtuner list-cfg
```

能列出内置配置文件，说明 CLI 可用

---

## 4\. 下载模型

XTuner 支持 HuggingFace / ModelScope / OpenXLab 多个模型源，本篇用 **ModelScope** 。

```
from modelscope import snapshot_download

# 下载 Qwen/Qwen2.5-1.5B-Instruct
model_dir = snapshot_download(
    'Qwen/Qwen2.5-1.5B-Instruct',
    cache_dir='/root/autodl-tmp/model/'
)

print(f"模型已下载到：{model_dir}")
```

下载完成后，模型路径：

```
/root/autodl-tmp/model/Qwen/Qwen2.5-1.5B-Instruct
```

后续配置文件中 `pretrained_model_name_or_path` 填这个路径。

---

## 5\. 准备微调数据

XTuner 默认支持 **Alpaca 格式** （与 LLaMA-Factory 一致）：

```
[
  {
        "conversation": [
            {
                "input": "马上要上游泳课了，昨天洗的泳裤还没干，怎么办",
                "output": "游泳时泳裤本来就会湿，不用晾干。"
            }
        ]
    }
]
```

把文件命名为 `target_data.json` ，放在自己的数据目录下：(完整数据集，可以问我要)

```
mkdir -p /root/dataset/xtuner
# 上传或拷贝 target_data.json 到此目录
ls /root/dataset/xtuner/target_data.json
```

---

## 6\. 微调配置文件（核心）

XTuner 的核心是 **一份 Python 配置文件** ——所有训练参数都写在这里。

### 6.1 复制官方模板

```
# 查看所有内置配置
xtuner list-cfg

# 复制一个接近的 Qwen QLoRA 模板到当前目录
cd /root
xtuner copy-cfg qwen1_5_1_8b_chat_qlora_alpaca_e3 .

# 复制后会生成 *_copy.py，改个更直观的名字
mv qwen1_5_1_8b_chat_qlora_alpaca_e3_copy.py qwen2_5_1_5b_instruct_qlora_alpaca_e3.py
```

官方提供了几十个预置模板，覆盖 Qwen / InternLM / Llama 等主流模型 + LoRA / QLoRA / 全参微调。用 `xtuner list-cfg | grep qwen` 可快速筛选 Qwen 相关配置。

### 6.2 修改配置（PART 1：路径与超参数）

打开 `qwen2_5_1_5b_instruct_qlora_alpaca_e3.py` ， **只需要改 PART 1** ：

```
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/root/autodl-tmp/model/Qwen/Qwen2.5-1.5B-Instruct'

# Data
data_files = '/root/dataset/xtuner/target_data.json'
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 512

# parallel
sequence_parallel_size = 1# 序列并行大小，1 = 不开

# Scheduler & Optimizer
batch_size = 2# per_device
accumulative_counts = 8
max_epochs = 10000
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1# grad clip
warmup_ratio = 0.03

# Save
save_steps = 100
save_total_limit = 2# 最多保留几个 checkpoint

# Evaluate the generation performance during the training
evaluation_freq = 100
SYSTEM = SYSTEM_TEMPLATE.alpaca
evaluation_inputs = [
    '只剩一个心脏了还能活吗？',
    '爸爸再婚，我是不是就有了个新娘？',
    '我只出生了一次，为什么每年都要庆生',
]
```

**关键参数说明：**

| 参数 | 含义 | 推荐值 |
| --- | --- | --- |
| `pretrained_model_name_or_path` | 基座模型绝对路径 | 上一节下载的路径 |
| `data_files` | 训练数据路径 | 自己的 JSON |
| `max_length` | 文本最大长度 | 512 / 1024 / 2048 |
| `batch_size` | 每卡 batch size | 显存够就调大（2~8） |
| `accumulative_counts` | 梯度累积 | 8~16（等效放大 batch） |
| `max_epochs` | 训练轮数 | 3~10 |
| `lr` | 学习率 | LoRA 用 1e-4 ~ 2e-4 |
| `sequence_parallel_size` | 序列并行大小 | 单机多卡保持 1 |

### 6.3 修改配置（PART 3：数据集）

```
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path="json", data_files=data_files),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,  # 数据已是 conversation 格式，无需额外转换
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn,
)
```

`dataset_map_fn=None` 表示用默认的 Alpaca 格式解析（与 LLaMA-Factory 一致）。

### 6.4 LoRA 与量化配置（PART 2，已内置）

模板里已经写好了 QLoRA 配置（4-bit 量化 + LoRA），通常 **不需要改** ：

```
# PART 2 中已经预置的 QLoRA 配置
model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,              # ← 4-bit 量化
            bnb_4bit_quant_type="nf4",      # ← NF4 量化类型
        ),
    ),
    lora=dict(
        type=LoraConfig,
        r=32,           # LoRA rank
        lora_alpha=64,  # 缩放系数
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)
```

如果想跑全量 LoRA（不用 4-bit 量化），把 `quantization_config` 整段删掉即可。

---

## 7\. 单卡微调与监控

配置改好后，用单卡启动训练：

```
export OMP_NUM_THREADS=1

xtuner train /root/qwen2_5_1_5b_instruct_qlora_alpaca_e3.py
```

训练过程中关注三件事： `evaluation_freq` 间隔打印生成样例 **（观察是否学到位）、** loss 平稳下降 **、** checkpoint 按时落盘。

XTuner 训练日志

单卡 checkpoint 为单个 `.pth` 文件，保存在 `work_dirs/<config_name>/` （在 `/root` 下执行即为 `/root/work_dirs/...`）：

```
work_dirs/qwen2_5_1_5b_instruct_qlora_alpaca_e3/
├── iter_100.pth
├── iter_200.pth
├── iter_300.pth
└── ...
```

后续转换直接使用 `.pth` 路径即可；训练中断续训见下一节。

---

## 8\. 中断继续训练

训练中途断了，可以改配置文件从 checkpoint 恢复：

```
# qwen2_5_1_5b_instruct_qlora_alpaca_e3.py
# load from which checkpoint
load_from = "/root/work_dirs/qwen2_5_1_5b_instruct_qlora_alpaca_e3/iter_100.pth"

# whether to resume training from the loaded checkpoint
resume = True  # ← 关键：设为 True 才真正续训
```

`resume=True` 会同时恢复 optimizer 状态、随机种子、step 计数等； `resume=False` 只加载模型权重（不推荐用于中断续训）。

改完后重新执行 `xtuner train` 命令即可续训。

---

## 9\. 模型转换：PTH → HuggingFace

XTuner 训练产物是 **PTH 格式** （PyTorch 原生），不能直接用 transformers 加载。需要转换为 **HuggingFace 格式** ：

```
xtuner convert pth_to_hf \
    /root/qwen2_5_1_5b_instruct_qlora_alpaca_e3.py \
    /root/work_dirs/qwen2_5_1_5b_instruct_qlora_alpaca_e3/iter_300.pth \
    /root/xtuner/work_dirs/hf
```

**参数说明：**

| 参数 | 含义 |
| --- | --- |
| 第 1 个 | 配置文件（必须与训练时一致） |
| 第 2 个 | 训练产物的 `.pth` 路径 |
| 第 3 个 | HF 格式输出目录 |

![[公众号文章/assets/一文跑通 XTuner：环境搭建、QLoRA 微调、VLLM部署全流程/c367e7f7f56d207e8a604331a150809f_MD5.webp]]

转换hf

转换完成后， `/root/xtuner/work_dirs/hf/` 目录下得到一份 HuggingFace 格式的 **LoRA adapter** （不是完整模型）。

---

## 10\. 模型合并：LoRA Adapter → 完整模型

HF 格式的产物 **只包含 LoRA adapter 权重** （几十 MB），需要合并回基座模型才能得到完整权重：

```
xtuner convert merge \
    /root/autodl-tmp/model/Qwen/Qwen2.5-1.5B-Instruct \
    /root/xtuner/work_dirs/hf \
    /root/xtuner/work_dirs/merged
```

**参数说明：**

| 参数 | 含义 |
| --- | --- |
| 第 1 个 `LLM` | **基座模型路径 （不是 adapter）** |
| 第 2 个 `LLM_ADAPTER` | 上一节转换的 HF adapter 目录 |
| 第 3 个 `SAVE_PATH` | 合并后完整模型的输出路径 |

![[公众号文章/assets/一文跑通 XTuner：环境搭建、QLoRA 微调、VLLM部署全流程/c95a33225e084184bb47b81ccbb62993_MD5.webp]]

合并完成后， `/root/xtuner/work_dirs/merged/` 是一份 **可直接部署** 的完整模型。

**合并阶段不需要 GPU 分布式** ——用单卡甚至 CPU 都能跑（速度慢些）。

---

## 11\. 合并后模型的使用

合并后的模型可以像普通 HuggingFace 模型一样，用 `transformers` 加载并做对话推理——加载路径指向 §10 合并输出目录 `/root/xtuner/work_dirs/merged` ，问题列表可与配置里 `evaluation_inputs` 保持一致，便于对比训练过程中的生成效果。

到这里， **单卡微调的主线已经走通** ：训练 → 转换 → 合并 → 本地推理。下面部署到 vLLM，方便对外提供 API 服务。

---

## 12\. 部署到 vLLM

合并后的模型可用 vLLM 对外提供 OpenAI 兼容 API。XTuner 训练时使用 `PROMPT_TEMPLATE.qwen_chat` ，部署时 **必须指定同一套对话模板** ，否则容易出现「微调正常、上线答非所问」。

```
# 查出与训练一致的 chat template 路径
python -c "import xtuner, pathlib; print(pathlib.Path(xtuner.__file__).parent / 'chat_templates' / 'qwen_chat.json')"

vllm serve /root/xtuner/work_dirs/merged \
    --chat-template <上一步输出的路径>
```

**详细说明请读第十二篇：**

《 [（十二）模型微调很成功，vLLM部署却"答非所问"，这个细节很多人不知道](https://mp.weixin.qq.com/s?__biz=MzYzNjI2NjMyNA==&mid=2247485721&idx=1&sn=ae6c962785276eeca0a4ba5a8aca1389&scene=21#wechat_redirect) 》——对话模板原理、正常/异常部署路径、测试脚本与常见报错，该篇已完整展开，本篇不再重复。

---

## 13\. 小结

本篇走通 XTuner **单卡** 配置驱动微调主线： **环境搭建 → 改配置 → 单卡训练 → 转换合并 → 推理 / vLLM 部署** 。

下一篇接着讲： **单机多卡、指定卡、多机多卡** ，以及常见问题排查。多卡时配置文件基本不用改，只需换启动命令；转换产物从单个 `.pth` 变成文件夹，细节见下篇。

---

**系列导航**

| 篇章 | 主题 |
| --- | --- |
| 上篇 | 分布式训练基本概念 + DeepSpeed 框架 |
| 中篇 | LLaMA-Factory 多卡训练实操 |
| **本篇** | **一文跑通 XTuner：环境搭建、QLoRA 微调、VLLM部署全流程** |
| 下篇 | XTuner 分布式微调 |

**微信扫一扫赞赏作者**

微调与部署 · 目录