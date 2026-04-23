## hermes Agent 实现 LLM Wiki知识库思想

| 维度            | 传统 RAG（无状态检索模式）                                                         | LLM WIKI（有状态编译模式）                                                     |
| :------------ | :---------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| **模式定位**      | 无状态检索                                                                   | 有状态编译                                                                 |
| **核心特点**      | 每次查询从零开始，知识不积累、不关联                                                      | 知识编译一次，持续更新，复利增长                                                      |
| **Step 1**    | **文档分块 (Chunking)**                                                     | **完整摄入 (Ingest)**                                                     |
|               | 原始文档按固定 token 数（如 512）机械切割为碎片，语义边界被打断                                   | LLM 完整读取源文档并理解全文语义，与用户讨论关键要点后提取结构化知识                                  |
| **Step 2**    | **向量嵌入 (Embedding)**                                                    | **知识编译 (Compile)**                                                    |
|               | 碎片经嵌入模型转为高维向量，存入向量数据库（Pinecone/Weaviate/Chroma）                         | 写入摘要页面，创建/更新实体和概念页面。一次摄入可能触及 10-15 个 Wiki 页面                          |
| **Step 3**    | **相似度检索 (Retrieval)**                                                   | **交叉引用 (Cross-reference)**                                            |
|               | 查询时计算余弦相似度，取 Top-K 碎片注入 LLM 上下文窗口                                       | `[[wikilink]]` 互联 + YAML frontmatter 标签。矛盾在摄入时主动标记 `[!contradiction]` |
| **Step 4**    | **临时生成 (Generation)**                                                   | **编译查询 (Query)**                                                      |
|               | LLM 从不相关联的碎片中临时拼凑答案，质量完全依赖检索命中率                                         | 读取 `index.md` 定位 → 深入相关页面 → 从结构化知识综合答案（非碎片拼凑）                         |
| **Step 5**    | **答案丢弃 (Discard)**                                                      | **知识归档 (Archive)**                                                    |
|               | 答案消失在聊天历史中，综合分析不可复用，下次查询重走全流程                                           | 有价值的答案归档回 Wiki 成为新页面，你的探索像摄入源一样在 Wiki 中复利增长                           |
| **本质问题/核心优势** | **本质问题**：知识是"一次性消费品"，每次查询都在重复发现相同的知识。问一个需要综合 5 篇文档的微妙问题，LLM 必须每次都找到相关碎片 | **核心优势**：Wiki 是一个持久化的、复利增长的知识资产。交叉引用已经建立，知识可复用、可演进                    |

![[LLM Wiki知识库展示.png]]
### LLM Wiki 完整工作流

| 步骤     | 环节                 | 说明                                 |
| :----- | :----------------- | :--------------------------------- |
| **01** | **Ingest 导入**      | 看到好文章直接丢进 `raw/`<br>不需手动分类、标注、整理   |
| **02** | **Compile 编译**     | LLM 自动提炼关键概念<br>撰写词条，建立页面间链接，更新索引  |
| **03** | **Query 查询**       | 提问时从 `wiki/` 读取结构化内容<br>不再反复扫描原始文档 |
| **04** | **Lint / Heal 自检** | 定期扫描整个 wiki<br>处理矛盾、孤立页面、过时内容      |
### Karpathy 真实工具栈：
![[AI应用/assets/LLM Wiki知识库/95d9f98c331928a7ce51c3b55df80648_MD5.png]]



### 创建 LLM wiki 示例的hermes过程日志
![[llm-wiki.png]]
```
+- 数据：dataset, preprocessing, augmentation
+- 机构：company, lab, open-source
+- 人物：researcher, engineer
+
+## Page Thresholds
+- 创建新页面：当实体/概念出现在 2+ 篇论文中，或是一篇论文的核心主题。
+- 避免为次要提及创建页面。
  ╎ ✍️  write     /root/wiki/index.md  0.6s
  ┊ review diff
a//root/wiki/index.md → b//root/wiki/index.md
@@ -0,0 +1,13 @@
+# Wiki Index
+
+> 大模型微调知识库目录
+> 最后更新：YYYY-MM-DD | 总页数：0
+
+## Entities
+<!-- 按字母顺序排列 -->
+
+## Concepts
+
+## Comparisons
+
+## Queries
  ╎ ✍️  write     /root/wiki/log.md  0.6s
  ┊ review diff
a//root/wiki/log.md → b//root/wiki/log.md
@@ -0,0 +1,5 @@
+# Wiki Log
+
+## [2026-04-22] create | 知识库初始化
+- 领域：大模型微调技术
+- 结构已创建：SCHEMA.md, index.md, log.md

╭─ ⚔ Ares ─────────────────────────────────────────────────────────────────────────────────────────────── ─────────────────────────────────────────────────────────────────────────────────────────────────╮
    已成功创建大模型微调论文知识库，结构如下：

    ```
    /root/wiki/
    ├── SCHEMA.md          # 知识库的领域定义、约定和标签分类
    ├── index.md          # 内容目录
    ├── log.md            # 操作日志
    ├── raw/papers/       # 存储原始论文文件
    ├── entities/         # 实体页面（模型、机构、人物等）
    ├── concepts/         # 概念页面（技术、方法等）
    ├── comparisons/      # 对比分析
    └── queries/          # 有价值的查询结果
    ```

    ### 下一步建议：
    1. **添加论文**：将论文 PDF 或 Markdown 文件放入 `/root/wiki/raw/papers/`，然后使用 `ingest` 流程提取信息。
    2. **查询知识库**：直接提问关于大模型微调的问题，我会从知识库中检索并合成答案。
    3. **定期维护**：运行 `lint` 检查知识库的一致性。

.......................... 此处省略我要求大模型 下载指定文章
 
//home/qyc/hermes_work/log.md → b//home/qyc/hermes_work/log.md
@@ -9,3 +9,10 @@
 - 领域：大语言模型训练与应用
 - 创建结构：SCHEMA.md, index.md, log.md
 - 目录：raw/{articles,papers,transcripts,assets}, entities, concepts, comparisons, queries
+
+## [2026-04-23] ingest | Think Deep, Not Just Long (arXiv 2602.13517)
+- 来源：https://arxiv.org/pdf/2602.13517
+- 保存原文：raw/articles/2602-13517-think-deep-not-just-long.md, raw/articles/2602-13517-think-deep-not-just-long.html
+- 创建实体页：entities/gpt-oss.md, entities/deepseek-r1.md, entities/qwen3.md
+- 创建概念页：concepts/deep-thinking-ratio.md, concepts/llm-overthinking.md, concepts/test-time-scaling.md, concepts/think-at-n.md, concepts/chain-of-thought.md
+- 更新：index.md (7 页面)
  ┊ 📋 preparing todo…
  ╎ 📋 plan      9 task(s)  0.0s

╭─ ⚔ Ares ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    论文已成功摄入知识库！

    **论文信息：**
    - 标题：Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens
    - 来源：arXiv 2602.13517 (2026-02-13)
    - 作者：Wei-Lin Chen, Liqian Peng, Tian Tan, et al. (University of Virginia, Google)

    **创建/更新的文件：**

    ```
    raw/articles/2602-13517-think-deep-not-just-long.md  (原文摘要)
    raw/articles/2602-13517-think-deep-not-just-long.html (HTML 原文)

    entities/gpt-oss.md
    entities/deepseek-r1.md
    entities/qwen3.md

    concepts/deep-thinking-ratio.md   (核心概念)
    concepts/llm-overthinking.md      (过度推理现象)
    concepts/test-time-scaling.md     (推理时扩展)
    concepts/think-at-n.md            (基于 DTR 的策略)
    concepts/chain-of-thought.md      (CoT 范式)

    index.md  (已更新，7 页)
    log.md    (已追加)
    ```
    **论文核心发现：**
    1. 输出 token 数与准确率负相关 (r = -0.544) -- 越长越差
    2. Deep-Thinking Ratio (DTR) 与准确率强正相关 (r = +0.828) -- 越 "深" 越好
    3. Think@n 策略：基于 DTR 早期筛选，在保持准确率的同时减少约 50% 推理成本
    要我继续摄入更多来源，或者对这个 wiki 做任何调整？
╰───────────────────────────────────────────────────────

........ 帮我准备一份这篇文章的汇报材料，要把文章做的实验，实验的目的和结论讲具体了

╭─ ⚔ Ares ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    汇报材料已全面重写，现在包含：

    **第一部分**：背景问题 -- 当前业界的 "越长越好" 假设为什么是错的

    **第二部分**：核心贡献概述

    **第三部分（重点）**：DTR 实验的完整细节
    - DTR 的定义和计算方法（JSD、收敛深度等）
    - 8 个模型变体 × 4 个基准的实验设置
    - 6 种基线方法的对比
    - 完整的相关系数数据表（含具体数值）
    - 超参数敏感性分析（g 和 ρ 的消融）
    - 替代距离度量验证（KL 散度、余弦相似度）

    **第四部分（重点）**：Think@n 实验的完整细节
    - 算法流程
    - 6 种对比方法的设计差异
    - GPT-OSS-120B-medium 和 Qwen3-4B-Thinking 的完整数据表
    - 前缀长度消融实验（50~2000 tokens 的对比）
    - 关键发现：50 token 前缀就能达到最佳效果

    **第五部分**：对我们的启示（分短期和中长期）
    **第六部分**：研究团队信息

    文件保存在：
    ```
    /home/qyc/hermes_work/queries/meeting-report-think-deep.md

```