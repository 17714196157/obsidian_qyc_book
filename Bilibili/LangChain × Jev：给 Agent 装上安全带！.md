

> [!info] 视频信息
> - **作者**：沧海九粟
> - **发布日期**：2026-09-21
> - **视频链接**：[Bilibili BV18Jhh6UEaZ](https://www.bilibili.com/video/BV18Jhh6UEaZ/)
> - **字幕语言**：中文

---

## 📖 简介

Agent 该用 Flash 还是 Pro？准备执行高风险操作时，如何提前拦截？

本期用 **LangChain × Jev** 搭建 Agent Harness，演示**模型自动路由**与 **Auto Mode 工具风险检查**。通过中英双语界面，直观看到模型选择、工具放行与拦截结果。

> [!tip] 快速开始
> ```bash
> # 安装 AgentSeek
> uv tool install agentseek
> 
> # 拉取模板
> agentseek create langchain/jev-harness --checkout main
> ```
> 按生成项目的 README 配置密钥并启动，即可跟着视频动手体验。

**相关资源**：
- AgentSeek：https://github.com/ob-labs/agentseek
- 模板仓库：https://github.com/agentseek-ai/agentseek-templates

---

## 🗂️ 章节导航

| 时间 | 内容 |
|------|------|
| `00:00` | 开场介绍 |
| `00:24` | Jev 模型特性 |
| `02:30` | LangChain 组件基础 |
| `05:06` | 应用场景：路由与自动模式 |
| `06:40` | Jev 注册与费用 |
| `07:38` | 代码实验演示 |

---

## 1. Jev 模型特性

> [!abstract] 核心定位
> Jev 是专注于**工具调用**与**结构化输出**场景的垂直优化模型，来自 **Type C AI**，被归类为 **System One 模型**。

### 关键优势

- ⚡ **更快**：比别人快将近 **200 倍**
- 💰 **更便宜**：价格仅为别人的 **1%**
- 🔄 **并行评估**：可同时评估多个问题并得出结构化结论

### 输入输出结构

| 组件 | 说明 |
|------|------|
| **State** | 上下文中的一小段状态描述，通常不长 |
| **Question** | 基于 State 提出的指引性问题，可多个并行 |
| **Answer Type** | 支持 `choice` / `score` / `now`（布尔值） |

> [!example] 示例：紧急程度判断
> - **State**：一段包含上下文的用户提问
> - **Question**：这条信息的紧急程度如何？
> - **Answer Type**：`now`（布尔值）
> - **结果**：`is_urgent = 0.999`（基本等于 1）

### 支持的答案类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `choice` | 从多个选项中选一个 | A / B / C / D |
| `score` | 0~1 区间的评分值 | 0.85 |
| `now` | 布尔值（0 或 1） | 0.999 ≈ true |

> [!note] 为什么 LangChain 快速支持？
> 这种结构化输出 + 并行评估的能力，天生适合 **LLM 评估评测** 和 **意图识别** 场景。质量好、价格低、速度快，是理想的评估模型。

---

## 2. LangChain 组件基础

LangChain 提供了封装：**`langchain-type-safe`**

### 核心工具：`TypeIClassifier`

封装了 `state` 和 `question` 的组合：

```python
# 布尔值类型
question = "这条信息是否紧急？"  # instruction 中说明返回布尔值
# 结果：0 或 1

# Choice 类型
question = "应该使用哪个模型？"
choices = ["快速模型", "强力模型"]

# Score 类型
question = "这条回复的质量如何？"
# 返回 0~1 的评分，可细分多个评分项
```

> [!tip] 并行能力
> 一次性可以问多个问题，每个问题可以有不同的答案类型（布尔值、选项、打分），模型会并行返回结果。这对于复杂场景的意图识别非常高效。

---

## 3. 应用场景：路由与自动模式

### 3.1 Model Routing（模型路由）

> [!question] 核心问题
> 什么时候用便宜快速的模型？什么时候用强力但昂贵的模型？

这是一个典型的 **Choice** 场景：

```python
question = "当前状态下应该使用哪种模型？"
choices = [
    "更省钱的模型（适合简单任务）",
    "更强力的模型（适合复杂推理）"
]
```

通过 Jev 快速判断，实现**动态模型路由**。

### 3.2 Auto Mode（工具自动模式）

> [!warning] 核心问题
> 工具执行前，是否需要人类审批？还是可以自动放行？

这是当前 Coding Agent 中非常常见的能力：

- ✅ **低风险工具** → 自动放行执行
- ⚠️ **高风险工具** → 拦截，等待人类审批

```python
# 接入示例（一句话搞定）
# 绑定需要判别的工具白名单
# 例如：一个代数工具
```

> [!success] 效果
> 利用 Jev 可以**又快又便宜**地完成工具风险判断，接入 LangChain 中间件后非常方便。

---

## 4. Jev 注册与费用

### 注册流程

1. 前往 Jev 官网注册
2. 进入 **Wait List**（约 1 天左右收到邮件，快的话当天）
3. 注册后直接赠送 **$5** 额度

### 费用情况

| 项目 | 价格 |
|------|------|
| **输入** | $0.042 / 百万 token（约 3 毛） |
| **输出** | 目前**免费** 🎉 |

> [!tip] 建议
> 现在输出免费，赶紧体验！用赠送的 $5 做一些快速实验，初步了解 Jev 模型的能力。

---

## 5. 代码实验演示

实验基于 **LangChain Type Safe** 以及两个预制的中间件构建。

### 获取方式

通过 AgentSeek 命令行获取和运行：

```bash
agentseek create langchain/jev-harness --checkout main
```

### 实验内容

#### 第一块：模型路由

- 准备两个模型：**快速模型**（如 Flash）和 **长推理模型**（如 Pro）
- 不同任务内容 → Jev 自动选择不同模型进行后续推理

#### 第二块：Auto Mode 工具审批

| 场景 | 工具 | 环境 | 预期结果 |
|------|------|------|----------|
| 删除生产备份 | `delete_backup` | Production | ❌ 拦截 |
| 清除过期内容 | `delete_backup` | Staging | ✅ 放行 |

### 演示流程

> [!example] 场景一：删除生产备份
> 1. 选择任务：删除生产备份
> 2. 选择强推理模型
> 3. 执行工具时 → **自动拦截**
> 4. 可查看 Jev 原始回答 + Auto Mode 中间件输出

> [!example] 场景二：Staging 环境清除过期内容
> 1. 选择任务：清除 Staging 过期内容
> 2. 自动选择**快速模型**
> 3. 执行工具时 → **放行**（相对安全）
> 4. 可查看风险概率识别 + 工具放行输出

> [!note] 模板提供
> 应用模板包含**完整前端 + 后端代码**，可结合代码研究，也方便二次开发或微调。

---

## 6. 总结与展望

> [!quote] 核心观点
> Jev 模型的出现，将推动更多大模型厂商在**垂类场景**（特别是工具调用、评估评测、结构化输出）推出更快、更好、更便宜的模型。

开发者可以在以下场景更好地驾驭工具：

- 🎯 **意图识别**
- 🔧 **工具调用的自主判断**
- 📊 **LLM 评估评测**
- 🚦 **模型路由**
- 🛡️ **Agent 安全拦截**

---

## 🔗 相关链接

- [AgentSeek GitHub](https://github.com/ob-labs/agentseek)
- [AgentSeek 模板仓库](https://github.com/agentseek-ai/agentseek-templates)
- [Bilibili 视频](https://www.bilibili.com/video/BV18Jhh6UEaZ/)

---

## 📝 标签

#LangChain #Jev #Agent #工具调用 #结构化输出 #模型路由 #AutoMode #意图识别 #LLM评估 #AgentSeek