---
title: "第三讲：斜杠命令大全——像老手一样操控Hermes【Hermes入门11讲】"
source: "https://mp.weixin.qq.com/s/uDtExp6ARasXYBLoWfwgSQ"
author:
  - "[[安守正]]"
published:
created: 2026-05-25
description: "用了一个月Hermes之后，我觉得这东西值得好好写一写。它不是那种&quot;用完即走&quot;的AI聊天工具。"
tags:
  - clippings
  - hermes
  - cli
  - slash-commands
  - tutorial
---

> [!abstract] 来源信息
> 原创 安守正 *2026年5月20日 18:30*

![[公众号文章/assets/第三讲：斜杠命令大全——像老手一样操控Hermes【Hermes入门11讲】/47befbdd4da55616c6d6bd32f75a1412_MD5.webp]]

## 概述

用了一个月 Hermes 之后，我觉得这东西值得好好写一写。

它不是那种"用完即走"的 AI 聊天工具。它会记住你、能动手干活、还能自己长进。每天早上自动给我推 AI 新闻摘要，中午给我推市场建议，晚上给我推他自己一天给服务器做的维护和巡检报告，相当于一个**自主全能的秘书**。

但网上关于 Hermes 的中文资料太少了，官方文档又全是英文，而且对零基础小白不友好。我从 5 月 18 日开始，把自己摸索出来的经验整理成这个系列，从零开始手把手教。

如果你也想有个真正能干活的 AI 助手，而不是只会聊天的机器人，跟着往下看就行。

> [!tip] 作者心得
> 刚用 Hermes 的时候，每次想干点啥都要打一大段话描述。后来才发现输个 `/` 就能调出命令菜单，很多操作一条命令就搞定，快太多了。

---

## Hermes斜杠命令

在 Hermes 的对话界面里，输入 `/` 会弹出一个自动补全菜单，列出所有可用的命令。命令不分大小写，`/HELP` 和 `/help` 效果一样。

装了的技能也会自动变成斜杠命令，比如装了 `plan` 技能后，输 `/plan` 就能调用。

## 1. 基础操作类

### 1.1 `/new` 和 `/reset` —— 清屏重来

**什么时候用**：当前对话聊跑偏了、上下文太乱、或者想换个话题。

```
❯ /new
```

**效果**：清空当前对话上下文，保留配置和记忆，相当于开了个新对话但不用退出重进。`/reset` 和 `/new` 效果一样，随便用哪个。

**使用场景示例**：

```
❯ 帮我分析这段代码
（Hermes分析了一堆）
❯ /new
❯ 好了换件事，帮我写个邮件
```

### 1.2 `/title` —— 给会话起个名字

**什么时候用**：这个会话你想留着以后恢复，起个名字好找。

```
❯ /title 用户登录功能开发
```

之后恢复：

```
hermes -c "用户登录功能开发"
```

### 1.3 `/save` —— 手动保存会话

**什么时候用**：做了重要操作，想确保不会丢。

```
❯ /save
```

默认 Hermes 退出时自动保存，但关键时刻手动存一下更安心。

### 1.4 `/quit` —— 退出

```
❯ /quit
```

退出前会自动保存会话，并显示恢复命令。

---

## 2. 模型切换类

### 2.1 `/model` —— 查看和切换模型

**什么时候用**：当前模型不够用了（比如要处理复杂代码，想换个更强的），或者当前模型挂了。

```
❯ /model                                        # 查看当前模型配置
❯ /model anthropic/claude-opus-4                # 直接切换到Claude Opus
❯ /model                                        # 不加参数进入交互式菜单
```

**实际场景**：

```
❯ 帮我写个简单的Python脚本
（DeepSeek很快写完）
❯ /model anthropic/claude-opus-4
❯ 这个脚本有个边界情况我没想清楚，你深入分析一下
（切换到Opus处理复杂推理）
❯ /model deepseek/deepseek-chat
❯ 好了换回去，太贵了
```

---

## 3. 工具和信息类

### 3.1 `/tools` —— 查看当前可用的工具

**什么时候用**：想知道 Hermes 现在能干什么，或者排查某个功能为什么没反应。

```
❯ /tools
```

会列出所有当前启用的工具，比如 `terminal`、`web_search`、`read_file` 等。如果你发现 Hermes 不会搜索网页，先 `/tools` 看看 `web_search` 在不在列表里。

### 3.2 `/memory` —— 查看记忆内容

**什么时候用**：好奇 Hermes 记住了你什么，或者想确认某条信息有没有被记进去。

```
❯ /memory
```

显示当前 `MEMORY.md` 和 `USER.md` 的内容，带占用比例。

### 3.3 `/compress` —— 压缩上下文

**什么时候用**：聊了很久，状态栏变橙色/红色了，或者 Hermes 开始忘事。

```
❯ /compress
```

Hermes 会把中间部分的历史对话总结成摘要，腾出空间。经常用于处理复杂任务聊到一半的时候。

### 3.4 `/usage` —— 查看 Token 和花费统计

**什么时候用**：想知道这回合花了多少 token、总共花了多少钱。

```
❯ /usage
```

显示详细的输入/输出 token 数、预估花费。

### 3.5 `/insights` —— 使用统计

**什么时候用**：想看看最近用了多少、花在哪些模型上。

```
❯ /insights --days 7        # 最近7天
❯ /insights --days 30       # 最近30天
```

---

## 4. 人格和风格类

### 4.1 `/personality` —— 切换人格

**什么时候用**：想让 Hermes 换个说话风格，或者需要它在某种特定模式下工作。

**内置人格**：

| 人格 | 效果 | 使用场景 |
|------|------|----------|
| `helpful` | 标准助手模式 | 默认 |
| `concise` | 极简回复，不说废话 | **最常用**，默认太啰嗦 |
| `technical` | 技术风格，偏硬核 | 写代码时 |
| `teacher` | 教学风格，解释很细 | 学新东西时 |
| `creative` | 创意发散 | brainstorm 时 |
| `pirate` | 海盗风格，挺搞笑的 | 无聊时试试 |
| `kawaii` | 可爱风格 | 心情好时 |
| `shakespeare` | 莎士比亚风格 | 基本不用 |

```
❯ /personality concise      # 切到极简模式
❯ /personality technical    # 切到技术模式
```

> [!tip] 自定义人格
> 在 `~/.hermes/config.yaml` 里添加：
> ```yaml
> personalities:
>   暴躁老哥: "你是个脾气火爆但技术很强的程序员，回复简短直接，偶尔吐槽"
> ```
> 然后 `/personality 暴躁老哥` 就能用。

### 4.2 `/reasoning` —— 调整推理深度

**什么时候用**：复杂任务需要更深思考，或者简单任务想让它直接给答案别墨迹。

```
❯ /reasoning              # 查看当前设置
❯ /reasoning high         # 深度推理（质量好但慢、贵）
❯ /reasoning none         # 不推理，直接给答案（快、便宜）
❯ /reasoning show         # 显示模型的思考过程
❯ /reasoning hide         # 隐藏思考过程
```

**实际场景**：

```
❯ /reasoning high
❯ 帮我设计一个数据库表结构，要考虑未来可能的扩展
（深入分析各种方案）
❯ /reasoning none
❯ 好了，就按第一个方案，给我建表SQL
（直接输出，不废话）
```

---

## 5. 技能类

### 5.1 `/skills` —— 管理技能

```
❯ /skills                           # 列出已安装技能
❯ /skills browse                    # 浏览技能中心
❯ /skills search kubernetes         # 搜索技能
❯ /skills install openai/skills/k8s # 安装技能
```

### 5.2 `/技能名` —— 调用技能

装了技能后，技能名自动变成斜杠命令：

```
❯ /plan 帮我设计一个用户认证系统的方案
❯ /github-pr-workflow 给这个功能创建个PR
```

---

## 6. 会话管理类

### 6.1 `/checkpoint` 和 `/restore` —— 检查点

**什么时候用**：准备做一批可能有风险的操作，先存个检查点，出错了能回滚。

```
❯ /checkpoint               # 创建检查点
（做一堆操作）
❯ /restore                  # 回滚到最近的检查点
```

**实际场景**：

```
❯ /checkpoint
❯ 帮我重构这个项目的目录结构
（Hermes一顿操作）
❯ 呃不对，这个结构不太对
❯ /restore                  # 回到重构前的状态
```

### 6.2 `/busy` —— 控制打断行为

**什么时候用**：Hermes 正在干活的时候，你想发新消息但不想打断它。

```
❯ /busy status              # 查看当前模式
❯ /busy interrupt           # 新消息打断当前任务（默认）
❯ /busy queue               # 新消息排队，等当前任务完再处理
❯ /busy steer               # 新消息作为提示注入当前任务，不打断
```

**实际场景**：

```
❯ 帮我分析这个日志文件
（Hermes开始分析，很长）
❯ /busy queue
❯ 对了，分析完后顺便看看有没有重复的错误模式
（这条消息不会打断分析，等分析完再处理）
```

---

## 7. 显示控制类

### 7.1 `/verbose` —— 切换输出详细程度

**什么时候用**：想看 Hermes 具体调了哪些工具、传了什么参数，或者反过来想清静点。

```
❯ /verbose                  # 在 off → new → all → verbose 之间循环切换
```

| 模式 | 显示内容 |
|------|----------|
| `off` | 只显示最终结果 |
| `new` | 只在切换工具时显示一行 |
| `all` | 显示每个工具调用（默认） |
| `verbose` | 显示完整参数和调试信息 |

> [!note] 作者习惯
> 日常用 `all`，出问题调 `verbose`。

### 7.2 `/skin` —— 切换界面皮肤

```
❯ /skin                     # 查看当前皮肤
❯ /skin 皮肤名               # 切换皮肤
```

---

## 8. 语音类

### 8.1 `/voice` —— 语音模式

```
❯ /voice on                 # 开启语音输入
❯ /voice off                # 关闭
❯ /voice tts                # 开关语音播报回复
```

开了 `/voice on` 后，按 `Ctrl+B` 录音，再按一下停止。Hermes 自动转文字处理。

---

## 9. 后台任务类

### 9.1 `/background` —— 后台执行

**什么时候用**：有个任务要跑很久，但你想在前台继续聊别的。

```
❯ /background 分析一下/var/log目录下的日志，找出今天的错误
```

Hermes 会在后台开一个新会话执行这个任务，前台你可以正常聊天。任务完成后结果会弹出来。

**实际场景**：

```
❯ /background 把项目里所有TODO找出来，按文件分类
❯ （继续在前台聊别的）
（几分钟后弹出后台任务结果）
```

---

## 10. 其他实用命令

| 命令 | 说明 |
|------|------|
| `/platforms` | 显示当前配置的平台信息 |
| `/footer` | 开关消息末尾显示模型/Token/时长信息 |
| `/reload-mcp` | 修改 MCP 配置后，不用重启 Hermes，直接重载 |

```
❯ /platforms                # 查看平台状态
❯ /footer                   # 切换运行时信息页脚
❯ /reload-mcp               # 重载MCP服务
```

---

## 11. 自定义快捷命令

有些操作你经常做，可以在 `~/.hermes/config.yaml` 里定义快捷命令：

```yaml
quick_commands:
  status:
    type: exec
    command: df -h /
  gpu:
    type: exec
    command: nvidia-smi
  update:
    type: exec
    command: cd ~/.hermes/hermes-agent && git pull
```

然后输 `/status` 直接看磁盘，`/gpu` 看显卡，`/update` 更新 Hermes。这些不走 AI，本地直接执行，**零 token**。

---

## 12. 作者日常 Workflow

分享几个实际在用的操作流：

> [!example]- 开始一天工作
> ```
> ❯ /personality concise
> ❯ /reasoning high
> ❯ /title 今天的工作
> ```

> [!example]- 聊久了上下文满了
> ```
> ❯ /compress
> ```

> [!example]- 遇到复杂问题换强模型
> ```
> ❯ /model anthropic/claude-opus-4
> （处理完）
> ❯ /model deepseek/deepseek-chat
> ```

> [!example]- 做风险操作前
> ```
> ❯ /checkpoint
> （让Hermes改代码）
> ❯ 嗯好像还行，不用restore了
> ```

> [!example]- 长任务不耽误聊天
> ```
> ❯ /background 分析一下过去一周的日志
> ❯ （继续在前台讨论别的）
> ```

---

## 13. 命令速查表

| 命令 | 作用 | 使用频率 |
|------|------|----------|
| `/new` | 新开对话 | ⭐⭐⭐⭐⭐ |
| `/title` | 命名会话 | ⭐⭐⭐⭐ |
| `/model` | 切换模型 | ⭐⭐⭐⭐ |
| `/personality` | 切换人格 | ⭐⭐⭐⭐ |
| `/reasoning` | 调整推理 | ⭐⭐⭐ |
| `/compress` | 压缩上下文 | ⭐⭐⭐⭐⭐ |
| `/memory` | 查看记忆 | ⭐⭐⭐ |
| `/usage` | 查看花费 | ⭐⭐⭐ |
| `/tools` | 查看工具 | ⭐⭐ |
| `/skills` | 查看技能 | ⭐⭐⭐ |
| `/background` | 后台任务 | ⭐⭐⭐ |
| `/checkpoint` | 创建检查点 | ⭐⭐⭐ |
| `/busy` | 控制打断 | ⭐⭐ |
| `/verbose` | 切换详细度 | ⭐⭐⭐ |
| `/voice` | 语音模式 | ⭐⭐ |
| `/quit` | 退出 | ⭐⭐⭐⭐ |
