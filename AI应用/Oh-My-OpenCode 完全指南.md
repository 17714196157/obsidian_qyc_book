---
title: "玩转 OpenCode(七)：Oh-My-OpenCode 完全指南"
source: "https://mp.weixin.qq.com/s/pz9pi1DMkNm5fHTplWOjKA"
author:
  - "[[翟星人的实验室]]"
published:
tags:
  - "clippings"
---

## 写在前面

前几篇一直介绍如何使用用 OpenCode配合AgentSkill 搭建了一个本地的超级 AI 助手，能处理文档、生成图片、制作视频、管理知识库。玩法其实还有很多，大家可以自行探索，本篇开始会介绍Opencode各种的高级用法，会有点难度，但不怂，就是干！单纯的opencode本身自带诸多基础agent能力，但用久了会发现，它就是单agent决策，有时候力不从心：

> 让它做前端，审美可能不太行
> 
> 让它调 Bug，容易原地打转
> 
> 让它搜文档，可能漏掉关键信息
> 
> 让它干到一半，它觉得差不多了就停了
> 
> 这时候需要的不是一个更强的 AI，而是一支 AI 团队。

那这篇介绍的 **Oh-My-OpenCode（OMO）** ，就是让 OpenCode 从单兵作战升级为团队协作的工具。让我们的智能体能力更加的多态！

---

## 一、OMO 是什么

Oh-My-OpenCode 是一个多智能体编排框架。简单说，就是让 AI 从一个人干活变成一个团队干活。

![[file-20260304155328873.webp]]

你可以把自己想象成一个 AI Manager，而智能体就是你的 Dev Team Lead。你只需要说清楚要做什么，团队会自己分工协作。

### 核心角色：Sisyphus

OMO 的主智能体叫 **Sisyphus（西西弗斯）** 。希腊神话里，西西弗斯被罚永远推石上山。这个名字的寓意是：它会不断推动任务，直到完成为止。

有个用户说过一句话：

> "If Claude Code does in 7 days what a human does in 3 months, Sisyphus does it in 1 hour."

虽然有点夸张，但确实能感受到效率的提升。

### 设计理念

OMO 的设计理念比较激进：

| 原则 | 含义 |
| --- | --- |
| 人类干预 = 失败信号 | 如果需要手动修复 AI 的代码，那是 AI 的失职 |
| 代码不可区分 | AI 写的代码应该和高级工程师写的一样好 |
| Token 换生产力 | 多用 Token 换取效率提升是值得的 |

第三点特别值得说一下。很多人担心多智能体会消耗更多 Token，但如果能省掉几小时的工作时间，多花点 Token 其实很划算。

---

## 二、智能体团队

OMO 内置了 10 多个专业智能体，每个都有自己的专长。

![[file-20260304155328866.webp]]

### 核心智能体

先看几个最常用的。

**Sisyphus** 是主编排器，负责规划、委派、执行。它的特点是不完成不停止，会一直推动任务直到真正做完。

**Oracle** 是架构顾问，只做分析、诊断、建议，不动手写代码。遇到架构问题、性能瓶颈、技术选型时可以问它。注意它是只读的，不会改你的代码。

**Librarian** 是文档专家，负责查官方文档、找开源实现、搜索代码。用一个新库不知道怎么用？让它去查。

**Explore** 负责代码探索，在代码库里快速搜索。想找某个功能的实现？让它去翻。

**Multimodal-Looker** 负责视觉分析，能看 PDF、图片、图表。需要分析一张架构图或者看 PDF 文档时用它。

| 智能体 | 职责 | 特点 |
| --- | --- | --- |
| Sisyphus | 主编排器 | 规划、委派、执行，不完成不停止 |
| Oracle | 架构顾问 | 只读，只分析诊断，不动手 |
| Librarian | 文档专家 | 官方文档、开源实现、代码搜索 |
| Explore | 代码探索 | 快速搜索代码库 |
| Multimodal-Looker | 视觉分析 | PDF、图片、图表分析 |

### 规划智能体

还有一组专门做规划的智能体。

**Prometheus** 负责战略规划。它不会直接开始干活，而是先问你一系列问题，搞清楚需求后再生成详细计划。适合复杂任务。

**Metis** 是计划顾问，会识别计划里的隐藏意图、歧义、可能导致 AI 失败的点。

**Momus** 是计划评审，验证计划是否清晰、可验证、完整。

**Atlas** 负责任务协调，执行 Prometheus 生成的计划。

| 智能体 | 职责 | 特点 |
| --- | --- | --- |
| Prometheus | 战略规划 | 访谈模式，问清楚再做 |
| Metis | 计划顾问 | 识别隐藏风险 |
| Momus | 计划评审 | 验证计划质量 |
| Atlas | 任务协调 | 执行计划 |

### 怎么调用

调用智能体有两种方式。

第一种是直接 @提及：

```
问 @oracle 这个架构设计合不合理？
问 @librarian 这个库怎么用？
问 @explore 找一下用户认证相关的代码
```

第二种是通过 delegate\_task 工具：

```
delegate_task(agent="oracle", prompt="帮我分析这个设计")
delegate_task(agent="librarian", prompt="查一下 React Query 的最佳实践")
```

日常用 @提及就够了，更自然。

---

## 三、两种工作模式

OMO 提供两种工作模式，适应不同场景。

![[file-20260304155328872.webp]]

### Ultrawork 模式

这是懒人模式。不想细想、直接干的时候用它。

只需要在提示词里加一个 `ulw` ：

```
ulw 给我的 Next.js 应用添加用户认证
```

就这一句话，智能体会自动完成整个流程：

> 先探索代码库，理解现有的代码模式
> 
> 让 Librarian 去研究最佳实践
> 
> 按照你的代码规范实现功能
> 
> 用诊断和测试验证结果
> 
> 如果有问题就修，没问题才算完
> 
> 你不用深入思考，智能体会替你思考。

这个模式最大的特点是 **并行执行** 。多个智能体同时工作，比单个 AI 快很多。

### Prometheus 模式

这是精确控制模式。复杂任务或关键变更时用它。

按 Tab 键进入 Prometheus 模式。Prometheus 会像产品经理一样访谈你：

> 你想实现什么功能？
> 
> 有什么约束条件？
> 
> 技术选型有偏好吗？
> 
> 需要兼容现有的什么模块？

问清楚之后，它会生成一份详细的执行计划，存在`.sisyphus/plans/` 目录里。你可以先看看计划，觉得没问题再执行。

执行时输入 `/start-work` ，Atlas 会接管并按计划执行。

什么时候用这个模式？

> 多天/多会话的项目，需要保持上下文
> 
> 关键生产变更，不能出错
> 
> 跨多文件的复杂重构
> 
> 需要留下决策记录，方便以后回溯

---

## 四、命令系统

OMO 内置了 6 个命令，每个都有特定的使用场景。

![[file-20260304155328874.webp]]

### /init-deep

这个命令用来生成代码库知识图谱。它会在每个目录下自动生成 AGENTS.md 文件，记录该目录的职责、文件说明、使用规范。

```
/init-deep
```

执行后，项目结构会变成这样：

```
project/
├── AGENTS.md              # 项目级说明
├── src/
│   ├── AGENTS.md          # src 目录说明
│   └── components/
│       └── AGENTS.md      # 组件目录说明
```

这些 AGENTS.md 文件会帮助 AI 更好地理解代码库结构。接手一个新项目时，先跑一遍这个命令，后续工作会顺畅很多。

大型项目可以限制深度：

```
/init-deep --max-depth=3
```

### /ralph-loop

这是 OMO 最核心的命令： **不完成不停止** 。

普通的 AI 经常会说"差不多了"然后停下来，但其实还没做完。ralph-loop 解决的就是这个问题。

工作机制：

> 智能体持续向目标工作
> 
> 自动检测是否真正完成
> 
> 如果中途想停，强制继续
> 
> 直到完成或达到最大迭代次数（默认100次）

使用示例：

```
# 实现完整功能
/ralph-loop "实现完整的用户认证系统，包含注册、登录、JWT、刷新令牌"

# 修复所有问题
/ralph-loop "修复所有 TypeScript 类型错误"
```

这个命令特别适合那种"必须做完"的任务。

### /ulw-loop

这是 ralph-loop 和 ultrawork 的组合， **最大火力输出** 。

既有 ultrawork 的并行能力，又有 ralph-loop 的不放弃精神。

```
# 全自动搭建项目
/ulw-loop "从零搭建一个 Next.js 电商网站，包含商品列表、购物车、结算"

# 大规模重构
/ulw-loop "将整个项目从 JavaScript 迁移到 TypeScript"
```

大型任务首选这个命令。

### /cancel-ralph

想停下来的时候用这个：

```
/cancel-ralph
```

也可以直接说自然语言：停止、取消、别干了。AI 能听懂。

### /refactor

智能安全重构。

重构代码最怕改坏东西。这个命令用 LSP 和 AST-grep 进行精确重构，每改一步都会验证。

```
# 重构指定模块
/refactor auth --scope=module

# 重构单个文件
/refactor src/utils/helpers.ts --scope=file
```

安全机制很严格：

> 每步变更后自动运行 lsp\_diagnostics 检查
> 
> 每步变更后自动运行测试
> 
> 发现问题立即停止，不会继续破坏

### /start-work

执行 Prometheus 生成的计划。

```
# 执行最近生成的计划
/start-work

# 从某个任务开始（跳过已完成的）
/start-work --from=task-3
```

这个命令要配合 Prometheus 模式使用。先按 Tab 做规划，确认计划后再执行。

### 命令速查

| 命令 | 一句话说明 | 典型场景 |
| --- | --- | --- |
| `/init-deep` | 生成代码库知识图谱 | 接手新项目 |
| `/ralph-loop` | 不完成不停止 | 实现完整功能 |
| `/ulw-loop` | 最大火力不停止 | 大型任务 |
| `/cancel-ralph` | 停下来 | 不想继续了 |
| `/refactor` | 安全重构 | 改代码不敢瞎改 |
| `/start-work` | 执行计划 | Prometheus 规划后 |

---

## 五、魔法关键词

除了命令，OMO 还支持魔法关键词。只要在提示词里包含这些词，就能激活对应模式。

![[file-20260304155328865 1.webp]]

最常用的是 `ulw` ， 激活 Ultrawork 模式：

```
ulw 修复所有 TS 错误
```

这比打 `/ulw-loop` 更快，适合临时任务。

其他几个魔法词：

| 关键词 | 效果 |
| --- | --- |
| `ultrawork`  / `ulw` | 最大并行，后台任务，深度探索 |
| `ultrathink` | 深度思考模式，适合复杂问题 |
| `search`  / `find` | 增强搜索模式 |
| `analyze`  / `investigate` | 深度分析模式 |

最简单的用法： `ulw` \+ 你的需求，然后喝杯咖啡。

```
ulw 帮我实现一个完整的用户认证系统，包含注册、登录、JWT、刷新令牌
```

---

## 六、内置技能

OMO 内置了 3 个核心技能，会在相关场景自动激活。

![[file-20260304155328864.webp]]

**playwright** 处理浏览器相关任务：自动化测试、截图、表单填写、网页爬取。

**frontend-ui-ux** 处理 UI/UX 设计任务。激活后 AI 会带着设计师人格，审美会好很多。

**git-master** 处理 Git 操作。这个技能有个特别的设计：原子提交。

### Git Master 的原子提交

很多人提交代码喜欢一次提一大堆文件，commit message 写个"fix bug"就完事了。这种提交历史后面很难看。

git-master 有一个核心原则： **多文件改动必须拆成多个提交** 。

```
3+ 文件 → 必须 2+ 个提交
5+ 文件 → 必须 3+ 个提交  
10+ 文件 → 必须 5+ 个提交
```

它还会自动检测仓库现有的提交风格（英文还是中文、什么格式），保持一致。

这样生成的提交历史，和人工仔细写的差不多，方便以后 review 和 bisect。

---

## 七、任务分类

OMO 支持按领域委派任务。不同分类会用不同的模型和参数，效果更好。

![[file-20260304155328861.webp]]

| 分类 | 用途 | 说明 |
| --- | --- | --- |
| `visual-engineering` | 前端、UI/UX、设计 | 用擅长视觉的模型 |
| `ultrabrain` | 复杂推理、架构决策 | 用推理能力强的模型 |
| `artistry` | 创意、设计 | 用创意能力强的模型 |
| `quick` | 快速简单任务 | 用便宜快速的模型 |
| `writing` | 文档、技术写作 | 用写作能力强的模型 |

使用方式：

```
delegate_task(category="visual-engineering", prompt="设计一个响应式导航栏")
delegate_task(category="ultrabrain", prompt="分析这个分布式系统的瓶颈")
```

不同的任务用不同的模型，既省钱又出效果。

---

## 八、内置 MCP 服务

OMO 内置了 3 个 MCP 服务器，不用额外配置就能用。

![[file-20260304155328860.webp]]

**websearch** 做实时网络搜索，用的是 Exa AI。需要查最新信息时会自动调用。

**context7** 查官方文档。用一个库不知道怎么用？它会去找官方文档。这个东西最实用，尤其是基于新技术框架去开发，大大提升效率。

**grep\_app** 搜 GitHub 代码。想看别人怎么实现某个功能？它会去 grep.app 搜开源代码。

这三个服务让 AI 能获取外部信息，不再局限于训练数据。

---

## 九、Hooks 系统

Hooks 是 OMO 的隐藏能力。它们在智能体生命周期的关键节点自动执行，你不用管，但它们在默默工作。

![[file-20260304155328865.webp]]

几个重要的 Hook：

**todo-continuation-enforcer** ：强制完成待办列表。AI 列了 5 个 todo，做了 3 个就想停？这个 Hook 会强制它继续。

**comment-checker** ：防止 AI 添加过多注释。有些 AI 喜欢加一堆注释，这个 Hook 会控制数量。

**keyword-detector** ：检测魔法关键词。你输入 `ulw` ，是这个 Hook 在检测并激活 Ultrawork 模式。

**session-recovery** ：会话崩溃自动恢复。万一中途断了，下次能接着上次的进度。

这些 Hook 默认全部启用。如果有些不想要，可以在配置文件里禁用：

```
{
  "disabled_hooks": ["comment-checker"]
}
```

---

## 十、LSP 和 AST 工具

OMO 内置了 IDE 级别的代码分析工具，这是它能精确重构的基础。这些做了解即可，基本都是agent自己触发。

![[file-20260304155328867.webp]]

### LSP 工具

LSP（Language Server Protocol）提供了类似 IDE 的能力：

- `lsp_diagnostics：获取错误和警告，不用跑构建就知道有没有问题`
- `lsp_goto_definition：跳转到定义`
- `lsp_find_references：查找所有引用`
- `lsp_rename：跨项目重命名符号`
- `lsp_symbols：获取文件大纲`

这些工具让 AI 能像人用 IDE 一样理解代码。

### AST-Grep 工具

AST-Grep 是基于语法树的搜索替换，比正则表达式精确得多。

- `ast_grep_search：AST 感知的代码搜索，支持 25 种语言`
- `ast_grep_replace：AST 感知的代码替换`

比如想把所有 `console.log` 换成 `logger.info` ， 用正则可能会误伤字符串里的内 容，AST-Grep 不会。

---

## 十一、安装与配置

### 安装

最简单的方式是让 AI 帮你装。把这段话复制给你的 OpenCode：

```
安装 oh-my-opencode，按照官方文档操作
```

它会自己去找文档、下载、配置。

如果想手动装，在 opencode.json 里添加：

```
{
  "plugin": ["oh-my-opencode@latest"]
}
```

### 配置文件

配置文件有两个位置：

- 项目级：`.opencode/oh-my-opencode.json`
- 用户级： `~/.config/opencode/oh-my-opencode.json`
- 项目级优先。大部分情况用默认配置就够了，想调整可以这样写：
	```
	{
	  "agents": {
	    "oracle": { "model": "openai/gpt-5.2" },
	    "librarian": { "model": "anthropic/claude-haiku-4-5" }
	  },
	  "categories": {
	    "quick": { "model": "anthropic/claude-haiku-4-5" }
	  }
	}
	```

可以给不同的智能体配不同的模型，平衡效果和成本。

---

## 十二、实际使用建议

用了一段时间 OMO 后，总结几个经验。

![[file-20260304155328859.webp]]

**小任务用 ulw，大任务用/ralph-loop或者 /ulw-loop**

临时改个 bug、加个小功能，直接 `ulw xxx` 就行。大功能、重构、迁移这种，用 `/ulw-loop` ，让它必须做完。

**复杂任务先规划**

如果任务涉及多个模块、需要做技术决策，先按 Tab 进 Prometheus 模式。花几分钟回答问题，生成的计划会更靠谱。

**不熟悉的项目先跑 /init-deep**

接手别人的项目，先跑一遍 `/init-deep` ，生成的 AGENTS.md 会帮 AI 理解代码结构。

**相信但验证**

OMO 很强，但不是万能的。重要的变更还是要 review 一下。好在它的输出质量通常不错，review 起来很快。毕竟ai不会给你背锅！

---

## 总结

Oh-My-OpenCode 把 OpenCode 从单个 AI 升级成了一个 AI 团队：

> 10+ 专业智能体，各司其职
> 
> Ralph Loop 让任务必须完成，不许半途而废
> 
> 魔法关键词，一个词激活整套能力
> 
> LSP、AST-Grep 让重构更安全
> 
> 25+ Hooks 在后台默默工作

核心用法就一句话：输入 ulw+ 你的需求，然后等结果。

试一下就知道效率提升有多明显。

---

以上，如果觉得有用，欢迎转发给需要的朋友。

我们，下篇见。

  

版权声明：  
本文由AI技术博客原创，转载请注明出处。

[#AI](https://mp.weixin.qq.com/s/) [#大模型](https://mp.weixin.qq.com/s/) [#ClaudeCode](https://mp.weixin.qq.com/s/) [#Opencode](https://mp.weixin.qq.com/s/) [#超级智能体](https://mp.weixin.qq.com/s/) [#AI办公自动化](https://mp.weixin.qq.com/s/) [#Agent](https://mp.weixin.qq.com/s/)

作者提示: 个人观点，仅供参考

继续滑动看下一个

翟星人的实验室

向上滑动看下一个