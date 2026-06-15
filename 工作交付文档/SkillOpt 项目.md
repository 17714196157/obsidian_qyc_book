SkillOpt是微软研究院推出的一个革命性框架，它改变了我们对AI Agent优化的理解。传统方法要么微调模型（成本高、不可迁移），要么手工调提示（不稳定、无系统）。SkillOpt提出了第三条路：把Agent的"技能"（skill.md）作为可训练的外部状态，通过类似深度学习的训练循环来优化它。
核心创新包括：文本学习率（限制每步编辑范围）、验证门控（只接受改进的编辑）、拒绝缓冲（将失败的编辑作为负反馈）和慢更新机制（长期稳定性）。在6个基准测试、7个模型、3种执行模式的全面评估中，SkillOpt在52个测试单元上全部获胜。

论文：https://arxiv.org/html/2605.23904v2
项目地址： https://github.com/microsoft/SkillOpt

# 尝试目标
```
论文：https://arxiv.org/html/2605.23904v2
项目地址： https://github.com/microsoft/SkillOpt
git项目我已经clone到本地/home/qyc/skillopt/SkillOpt了
论文已经下载好放在了 /home/qyc/skillopt/skillopt.pdf

任务：想用自己的数据集复现一下这个SkillOpt论文的思想。

具有要求：
1.需要创建一个跑这个SkillOpt项目代码的环境， conda create -yn SkillOpt python=3.11， 实现环境隔离
2.这个项目需要的数据格式是什么样子的，给我一个示例，让我可以将自己的数据集无缝对接是使用这个项目
3.所有操作和相关文件，更目录放在/home/qyc/skillopt


custom数据集具体说明：
自己的custom数据集放在 /home/qyc/skillopt/report_opt_ner.xlsx ，里面doc_id、content、output  三个字段， content是操作记录报告文本，output是参考答案，总样本数: 245，内容长度统计: 最小=91, 最大=491, 平均=241.43，content内容最大不过500，提示词也就1000。
我自己的数据集是一个操作名称识别的任务,我有一个初始化的提示词prompt模板
###
#角色：你是医疗专家。\n#任务:按照‘说明’，分析以下‘手术操作记录’内容，严格按照步骤执行。\n说明：\n1.各步骤结果独立返回，互不影响。\n2.各返回结果内如包含多个手术操作名称，统一以‘+’连接。\n3.对于手术室手术，如果记录中没有该手术的详细手术步骤，则认为当前手术记录并不是对该手术的记录。\n4.淋巴结穿刺术没有具体淋巴结，认为不规范；静脉穿刺没有具体静脉，认为不规范。\n步骤：\n第一步：判断该记录内容是否为消化道内镜检查、消化道内镜下治疗、下呼吸道内镜检查、下呼吸道内镜下治疗。如果‘是’，进入第二步；如果‘否’，进入第三步。\n第二步：选择该记录中实施的内镜检查名称，选项为：胃镜检查、胃-十二指肠镜检查、超声胃镜检查、食管镜检查、超声食管镜检查、小肠镜检查、超声内镜下十二指肠检查、超声内镜下小肠检查、结肠镜检查、纤维结肠镜检查、电子结肠镜检查、超声结肠镜检查、乙状结肠镜检查、气管镜检查、支气管镜检查、纤维支气管镜检查、超声支气管镜检查、电子支气管镜检查、硬质支气管镜检查。返回内镜检查名称，以‘名称1<answer>’ ‘ </answer>’包裹。如果不存在内镜下治疗，结束流程。如果存在内镜下治疗，进入第五步。\n第三步：识别该记录标题，从标题中抽取手术或操作名称。如果可以抽取到手术或操作名称，进入第四步；否则，进入第五步。\n第四步：判断抽取的手术或操作名称是否规范。如果‘规范’，返回从标题中抽取的手术或操作名称，以‘名称2<answer>’ ‘</answer>’包裹，流程结束；如果‘不规范’，不返回从标题中抽取的手术或操作名称，进入第五步。\n第五步：根据手术操作过程描述，完善手术或操作名称。返回该手术或操作名称，以‘名称3<answer>’ ‘</answer>’包裹。\n#输出格式：先给出结果，再解释。\n报告内容：###\n{{content}}\n###'
###
其中 content 变量是具体的有创操作记录报告文本内容。
大模型返回的结果，我会用 extract_answer_xml_操作名称抽取 python函数去解析一下，得到输出的唯一操作名称
'''python
def extract_answer_xml_操作名称抽取(text: str) -> str:
    # 尝试从文本中提取XML答案
    try:
        text_filter思考过程 = text.split("think")[-1]
        text = text_filter思考过程
        # pattern = r"<answer>(.*?)</answer>"
        pattern = "<answer>([\s\S]*?)</answer>"
        matches = re.findall(pattern, text)
        answer_list = []
        for match in matches:
            match = match.replace("...", "").strip()
            # print(match)
            answer_list.append(match)
        answer_list = [x.strip() for x in answer_list if x.strip() != ""]
        answer_list = [i for n, i in enumerate(answer_list) if i not in answer_list[:n]]
        answer_list = [x for x in answer_list if "think" not in x and "。" not in x]
        # answer_list = [x for x in answer_list if "无" not in x or "无痛" in x ]
        
        # return " + ".join(answer_list)
        """配合博士提示词的逻辑
        真正要的结果就是第一个输出，比如 输出了名称1+名称3，只取名称1；输出了名称2+名称3，只取名称2。
        识别出了输出中有名称1，后面就不用管了
        """ 
        answer_first = answer_list[0]
        answer_char = "<answer>"
        index_answer_char = answer_first.find(answer_char)
        if index_answer_char != -1:
          answer_first = answer_first[index_answer_char:]
          answer_first = answer_first.replace(answer_char,"")
        else:
          pass
        return answer_first
    # 捕获异常并打印错误信息
    except Exception as e:
        print(text)
        print(f"error: {traceback.format_exc()}")
        # input("!!!!")
        return ""
    
    return ""
'''

模型相关形象：
我的需要优化的大模型是 vllm部署的。
Target Model（学生）"http://192.168.0.181:8103/v1"    模型名称： "qwen2.5-14b-instruct" ,  密钥： "vllm-is-awesome"。
Optimizer 我希望配置成阿里百炼云服务上比较大的模型，阿里百炼云服务大模型配置如下： api_key =  "sk-d98a7434af1f4641921b8af02e175499"  base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  model_name="qwen3.6-plus"。
```

# **SkillOpt 医疗操作名称识别任务复现总结**。
### 📁 1. 目录结构与复现环境概览
整个项目位于 `/home/qyc/skillopt/` 下，分为**项目源码**、**自定义环境**、**数据**和**配置**四个部分。

```
text
/home/qyc/skillopt/
├── SkillOpt/                        # 🟦 官方源码 (git clone)
│   ├── scripts/
│   │   └── train.py                 # 🛠️ [修改] 添加强制注入逻辑 (混合架构)
│   └── skillopt/
│       ├── model/
│       │   └── qwen_backend.py      # 🛠️ [修改] 添加强制覆盖逻辑 (修复 URL 失效)
│       └── envs/
│           ├── base.py
│           ├── searchqa/
│           └── reportoptner/        # 🟢 [新增] 你的自定义医疗环境模块
│               ├── __init__.py
│               ├── adapter.py       # 适配 SkillOpt 训练循环
│               ├── dataloader.py    # 数据加载
│               ├── evaluator.py     # 自定义 XML 答案提取与 EM 评估
│               └── rollout.py       # Prompt 注入与 vLLM 输入截断保护
│
├── report_opt_ner.xlsx              # 📄 原始数据 (Excel, 245条)
├── convert_data.py                  # 🐍 数据转换脚本
├── test_llm.py                      # 🧪 连接测试脚本
│
├── custom_data/                     # 📂 处理后的数据目录
│   ├── train/items.json             # 训练集 (171条)
│   ├── val/items.json               # 验证集 (24条)
│   ├── test/items.json              # 测试集 (50条)
│   └── skills/
│       └── initial.md               # 初始 Skill 文件 (Prompt 模板)
│
└── configs/custom/
    └── reportoptner.yaml            # ⚙️ 训练配置文件 (混合架构)

```

### 🛠️ 2. 环境安装清单
```
bash
1.  **创建 Conda 环境**:
    conda create -yn SkillOpt python=3.11
    conda activate SkillOpt 
    
2.  **安装 SkillOpt 依赖**
    cd /home/qyc/skillopt/SkillOpt
    pip install -e .
    pip install pandas openpyxl  # 用于数据转换

3. **vLLM 部署 (前置条件)**:
在 `192.168.0.180:8103` 运行 `qwen2.5-14b-instruct` 模型。

4. **阿里云测试 qwen-plus 能使用(前置条件)**:
curl https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer sk-d98a7434af1f4641921b8af02e175499" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10
  }'
```

### 📊 3. 数据结构与转换
**原始数据 (`report_opt_ner.xlsx`)**
| doc_id | content (病历报告) | output (标准答案) |
| : | : | : |
| whslyy325269 | 检查名称：无痛肠镜...检查所见：循腔进镜... | 结肠镜检查 |
**转换后数据 (`custom_data/*/items.json`)**
适配 SkillOpt 的标准 JSON 格式，每个文件是一个 JSON 数组：
```
json
[
  {
    "id": "whslyy325269",
    "question": "请分析以下手术操作记录...",
    "context": "检查名称：无痛肠镜...检查所见：...",
    "answers": ["结肠镜检查"]
  }
]

```
*   **训练集 (171)**: 用于反思和技能进化。
*   **验证集 (24)**: 用于 Gate 验证，决定 skill 是否更新。
*   **测试集 (50)**: 最终测试集。

### 📝 4. 核心代码修改清单 (让项目跑通的关键)
为了让官方项目适配你的数据和硬件，我们进行了以下关键修改：
#### A. 新增自定义环境 (`skillopt/envs/reportoptner/`)
官方没有医疗环境的实现，我们需要从零搭建这个模块：
1.  **`evaluator.py`**: 复刻了你提供的 `extract_answer_xml` 逻辑，用于提取 `<answer>` 标签并计算准确率 (EM)。
2.  **`rollout.py`**:
    *   内置了你的 **5 步法 Prompt 模板**。
    *   添加了 `_truncate_content` (限制输入 6000 字) 和 `max_completion_tokens=2048`，**防止本地 vLLM 溢出**。
3.  **`adapter.py`**: 将上述模块接入 SkillOpt 的训练引擎。
#### B. 强制混合架构配置 (`scripts/train.py`)
**修改位置**: `main()` 函数中 `adapter = get_adapter(cfg)` 之前。
**目的**: 解决默认 URL 错误的问题，实现 **"小模型干活，大模型反思"**。
*   **Target (回答)**: 强制指向本地 `192.168.0.180:8103` (vLLM)。
*   **Optimizer (反思)**: 强制指向云端 `dashscope.aliyuncs.com` (阿里云)。
```python
def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    print(f"\n{'='*60}")
    print(f"  SkillOpt — Executive Strategy for Self-Evolving Agent Skills")
    print(f"{'='*60}")
    print(f"  env:            {cfg.get('env')}")
    print(f"  optimizer_model:  {cfg.get('optimizer_model')}")
    print(f"  target_model:  {cfg.get('target_model')}")
    print(f"  optimizer_backend:{cfg.get('optimizer_backend', 'openai_chat')}")
    print(f"  target_backend:{cfg.get('target_backend', 'openai_chat')}")
    print(f"  reasoning:      {cfg.get('reasoning_effort') or 'off'}")
    print(f"  rewrite_effort: {cfg.get('rewrite_reasoning_effort') or 'off'}")
    print(f"  epochs:         {cfg.get('num_epochs')}")
    print(f"  train_size:     {cfg.get('train_size') or 'from dataset'}")
    print(f"  steps/epoch:    auto")
    print(f"  batch_size:     {cfg.get('batch_size')}")
    print(f"  edit_budget:    {cfg.get('edit_budget')}")
    print(f"  lr_scheduler:   {cfg.get('lr_scheduler', 'constant')}")
    print(f"  update_mode:    {cfg.get('skill_update_mode', 'patch')}")
    print(f"  min_edit_budget:{cfg.get('min_edit_budget', 2)}")
    print(f"  minibatch_size: {cfg.get('minibatch_size')}")
    print(f"  seed:           {cfg.get('seed')}")
    print(f"  meta_skill:     {cfg.get('use_meta_skill', False)}")
    print(f"  skill_aware_reflection: {cfg.get('use_skill_aware_reflection', False)}")
    print(f"  slow_update:    {cfg.get('use_slow_update', False)}")
    print(f"  out_root:       {cfg.get('out_root')}")
	
    backend = cfg.get("model_backend") or cfg.get("backend") or ""
    if "qwen" in backend:
        # 必须在这里导入，确保拿到的是当前运行时的模块
        from skillopt.model.qwen_backend import (
            configure_qwen_chat,
            set_optimizer_deployment,
            set_target_deployment,
            TARGET_CONFIG,
            OPTIMIZER_CONFIG,
        )
        print("\n>>> [强制配置] 正在注入混合架构参数...")
        # 1. 强制覆盖环境变量 (最底层控制)
        os.environ["TARGET_QWEN_CHAT_BASE_URL"] = "http://192.168.0.180:8103/v1"
        os.environ["TARGET_QWEN_CHAT_API_KEY"] = "vllm-is-awesome"
        os.environ["TARGET_QWEN_CHAT_MAX_TOKENS"] = "2048"  # 🔥 限制 Target 输出，防溢出

        os.environ["OPTIMIZER_QWEN_CHAT_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["OPTIMIZER_QWEN_CHAT_API_KEY"] = "sk-d98a7434af1f4641921b8af02e175499"
        os.environ["OPTIMIZER_QWEN_CHAT_MAX_TOKENS"] = "32000" # ☁️ 放开 Optimizer 限制
        # 2. 调用配置函数，更新内存对象
        configure_qwen_chat(
            target_base_url="http://192.168.0.180:8103/v1",
            target_api_key="vllm-is-awesome",
            target_max_tokens=2048,
            optimizer_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            optimizer_api_key="sk-d98a7434af1f4641921b8af02e175499",
            optimizer_max_tokens=32000,
        )
        # 3. 强制指定模型名
        set_optimizer_deployment("qwen3.6-plus")
        set_target_deployment("qwen2.5-14b-instruct")
        # 4. 打印验证信息 (如果不显示正确的 URL，说明代码没跑通)
        print(f">>> [验证成功] Target URL   : {TARGET_CONFIG.base_url}")
        print(f">>> [验证成功] Optimizer URL : {OPTIMIZER_CONFIG.base_url}")
        print(f">>> [验证成功] Target Model  : {TARGET_CONFIG.deployment}")
        print(f">>> [验证成功] Optimizer Model: {OPTIMIZER_CONFIG.deployment}")
    # ==========================================
    adapter = get_adapter(cfg)

    # Build trainer and run
    from skillopt.engine.trainer import ReflACTTrainer
    trainer = ReflACTTrainer(cfg, adapter)
    summary = trainer.train()

    print(f"\n  Output saved to: {cfg['out_root']}")
    if summary.get("test_hard") is not None:
        print(f"  Final test: {summary['test_hard']:.4f}")
```
#### C. 底层兜底修正 (`skillopt/model/qwen_backend.py`)
**修改位置**: `_chat_messages_impl` 函数内部。
**目的**: 解决 Python 模块加载顺序导致的配置不生效问题。在发请求前强制覆盖 URL，确保请求绝对不会发错地方。

![[qwen_backend.png]]

#### D. Bug 修复 (`rollout.py`)
*   修复了 `diagnostic_trace_context_by_id` 为 `None` 导致的 `AttributeError`。
### ⚙️ 5. 核心配置说明 (`configs/custom/reportoptner.yaml`)
最终生效的配置采用了**混合架构**策略，既保证了速度又保证了长窗口需求：
| 组件 | 模型 | 部署位置 | 作用 | Max Tokens |
| : | : | : | : | : |
| **Target** | qwen2.5-14b-instruct | 本地 vLLM (180:8103) | 负责回答每个医疗样本 | **2048** (限制输出防溢出) |
| **Optimizer** | qwen3.6-plus | 阿里云 DashScope | 负责反思、分析失败案例 | **32000** (处理超长历史) |
### 🚀 6. 如何启动与监控
**启动命令**:

```
bash
conda activate SkillOpt
cd /home/qyc/skillopt/SkillOpt
python scripts/train.py --config /home/qyc/skillopt/configs/custom/reportoptner.yaml

```

**预期日志流**:
1.  **初始化**: 打印 Target 和 Optimizer 的正确 URL。
2.  **Rollout**: Target 模型处理样本，计算初始准确率 (Baseline)。
3.  **Reflect**: 将失败的样本历史发给 Optimizer，生成编辑建议。
4.  **Update**: 接受改进的建议，更新 Skill 文件。
5.  **Evaluate**: 在验证集上测试新 Skill 的准确率，如果提升则保留 (`ACCEPT`)。