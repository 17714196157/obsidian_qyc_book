---
title: "用 Hermes 打造 LLM Wiki 知识库：Karpathy 方法实战指南"
source: "https://mp.weixin.qq.com/s/QEymD9KVGiyhMf3rl1H_9w"
author:
  - "[[lusyoe]]"
published:
created: 2026-06-10
description:
tags:
  - clippings
  - hermes
  - llm-wiki
  - karpathy
  - 知识库
  - RAG
  - obsidian
---

> [!abstract] 来源信息
> 原创 lusyoe *2026年6月6日 16:14*

---

## 用 Hermes 打造 LLM Wiki 知识库：Karpathy 方法实战指南
把一堆文档扔给 AI，它帮你检索、回答问题。听起来很美，但用久了你会发现一个问题：**每次提问，AI 都在从头翻你的文档**。知识不会积累，问一次用一次，用完就丢。**2026 年 4 月，Karpathy 发了一个 GitHub Gist，提出了一个叫 **LLM Wiki** 的模式。核心思路很简单：别让 AI 每次都重新翻文档，让它**把知识编译成结构化的 wiki，持续维护**。

Hermes Agent 内置了一个 `llm-wiki` skill，把这个模式落地了。

### 1. RAG vs LLM Wiki

传统 RAG 的工作方式是"即时检索"：你问一个问题，系统从原始文档里找最相关的片段，拼进上下文，让 LLM 生成回答。

LLM Wiki 的思路完全不同：**知识先编译，再查询**。

```
                    ┌── 每次检索 ──→ 临时回答
原始文档 ── RAG ───┤
                    └── 知识不积累，用完就丢

                    ┌── 一次编译 ──→ 持久化知识页面
原始文档 ── Wiki ──┤                   ├── 交叉引用
                    └── 持续更新 ──→ + 持续更新
```

| 维度       | RAG        | LLM Wiki    |
| -------- | ---------- | ----------- |
| **工作方式** | 即时检索       | 先编译，再查询     |
| **知识状态** | 用完就丢       | 持久化，持续积累    |
| **回答来源** | 每次重新解析原始文档 | 直接读 wiki 页面 |
| **交叉引用** | 无          | 已做好，矛盾已标记   |
| **综合分析** | 每次临时生成     | 反映所有已收录来源   |

> [!quote] Karpathy 的比喻
> Obsidian 是 IDE，LLM 是程序员，wiki 是代码库。

### 2. 三层架构

LLM Wiki 的目录结构分三层，Hermes 的 `llm-wiki` skill 按这个思路设计了具体的组织方式：

```
wiki/
├── Layer 1: raw/              # 不可修改的原始材料
│   └── articles/              # 网页文章、PDF、会议记录等
│
├── Layer 2: wiki 页面          # Agent 维护的知识页面
│   ├── entities/              # 实体页面（工具、人物、产品）
│   ├── concepts/              # 概念页面（RAG 是什么、向量数据库原理）
│   ├── comparisons/           # 对比页面（Docker vs Podman）
│   └── queries/               # 有价值的查询结果归档
│
└── Layer 3: SCHEMA.md         # 定义结构和规则
    ├── index.md               # 知识索引
    └── log.md                 # 变更日志
```

#### 2.1 第一层：raw/

存放原始材料：网页文章、PDF 论文、会议记录等。**这些文件一旦存入就不再修改，是知识的"源代码"**。

#### 2.2 第二层：Wiki 页面

由 Agent 创建和维护的 markdown 文件，分成四种类型：

| 类型 | 目录 | 示例 |
|------|------|------|
| **实体页面** | `entities/` | 某个工具、某个人物、某个产品 |
| **概念页面** | `concepts/` | "RAG 是什么"、"向量数据库原理" |
| **对比页面** | `comparisons/` | "Docker vs Podman" |
| **查询归档** | `queries/` | 有价值的查询结果 |

#### 2.3 第三层：SCHEMA.md

Hermes 把这个配置文件命名为 `SCHEMA.md`（Karpathy 原始模式中叫 `CLAUDE.md` 或 `AGENTS.md`）。

> [!important] SCHEMA.md 的作用
> 定义了 wiki 的领域范围、命名规范、标签分类、页面创建门槛等。这是让 Agent 成为**"纪律严明的 wiki 维护者"**而不是**"随便聊天的 AI"**的关键。

---

### 3. 三大核心操作

#### 3.1 收录（Ingest）

把一篇新文章喂给 wiki，Agent 会：

1. 保存原始文件到 `raw/articles/`，计算 sha256 用于后续检测内容变化
2. 识别文章中提到的实体和概念
3. 检查 wiki 里是否已有相关页面，避免重复创建
4. 创建或更新 wiki 页面，添加交叉引用
5. 更新 `index.md` 索引和 `log.md` 日志

> [!tip] 知识复利
> 一篇来源文章可能触发 **5-15 个 wiki 页面的更新**。这就是"知识复利"的效果。

#### 3.2 查询（Query）

对 wiki 提问，Agent 会：

1. 先读 `index.md` 找到相关页面
2. 大型 wiki（100+ 页面）还会全文搜索
3. 读取相关页面后综合回答，**引用来源页面**
4. 如果回答有长期价值，归档到 `queries/` 目录

查询结果被归档后，下次遇到类似问题就不用重新推理了。

#### 3.3 健康检查（Lint）

定期让 Agent 检查 wiki 的健康状况：

| 检查项 | 说明 |
|--------|------|
| **孤儿页面** | 没有被其他页面引用的页面 |
| **断链** | wikilink 指向不存在的页面 |
| **过时内容** | 有新来源提及但页面超过 90 天没更新 |
| **矛盾标记** | `contested: true` 的页面需要人工审查 |
| **标签一致性** | 所有标签必须在 SCHEMA.md 的分类表里 |

---

### 4. 用 Hermes 实战

Hermes Agent 内置了 `llm-wiki` skill，直接用斜杠命令 `/llm-wiki` 就能激活完整的工作流。

#### 4.1 初始化 Wiki

```
/llm-wiki 帮我创建一个 wiki，领域是 AI 工具和开发技术。
```

Hermes 会创建目录结构，生成定制化的 `SCHEMA.md`、`index.md` 和 `log.md`。

> [!note] 自定义路径
> 在 `~/.hermes/.env` 中设置：
> ```
> WIKI_PATH=~/my-wiki
> ```
> 不设置的话默认在 `~/wiki`。

#### 4.2 收录文章

```
/llm-wiki 帮我把这篇文章收录到 wiki：https://example.com/some-article
```

Hermes 会用 `web_extract` 抓取内容，保存到 `raw/articles/`，然后自动创建或更新相关 wiki 页面。也可以一次性收录多篇文章，批量处理减少重复更新。

#### 4.3 提问

```
/llm-wiki 我的 wiki 里关于 Docker 的内容有哪些？
```

Hermes 会搜索 wiki，找到所有 Docker 相关页面，给出结构化的回答。

#### 4.4 健康检查

```
/llm-wiki 帮我检查一下 wiki 的健康状况。
```

Hermes 会运行完整的 lint 流程，报告问题并给出修复建议。

---

### 5. 和 Obsidian 配合

Wiki 目录就是一个普通的 markdown 文件夹，可以直接用 Obsidian 打开。

- `[[wikilinks]]` 在 Obsidian 里会渲染成可点击的链接
- **Graph View** 能直观看到知识网络的连接关系
- 如果装了 **Dataview 插件**，还能用 YAML frontmatter 做高级查询，比如"列出所有标签包含 `llm` 的页面"

---

### 6. 使用建议

| 建议                   | 说明                                           |
| -------------------- | -------------------------------------------- |
| **不要追求完美分类**         | 知识的边界是模糊的，先收录，交叉引用会帮你建立连接                    |
| **收录比查询更重要**         | wiki 的价值在于积累。读到好文章就"收录一下"，比用到再翻高效得多          |
| **定期健康检查**           | 每月跑一次 lint，清理断链和过时内容                         |
| **善用 confidence 标记** | 对只有一个来源支撑的观点标记 `confidence: medium`，避免弱证据当定论 |
|                      |                                              |

> [!quote] 知识管理的本质
> 不是收集信息，而是**让信息在需要的时候出现，并且是已经整理好的状态**。
>
> LLM Wiki 让这件事第一次变得真正可行。
