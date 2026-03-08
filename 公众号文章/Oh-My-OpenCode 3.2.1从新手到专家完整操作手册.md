---
title: "Oh-My-OpenCode 3.2.1从新手到专家完整操作手册"
source: "https://mp.weixin.qq.com/s/SrS0LxzDyC-0Dv9nbXUcYg"
author:
  - "[[码农不器]]"
published:
tags:
  - "clippings"
---
原创 码农不器 *2026年2月2日 17:18*

## OMO 3.2.1 从新手到专家完整操作手册

Oh-My-OpenCode（简称 OMO）是 OpenCode 的顶级插件，将单个 AI 编码代理升级为多代理协作团队。它提供专化代理（如 Sisyphus、Hephaestus、Oracle、Librarian 等）、并行执行、详细规划、工作流钩子等功能，帮助开发者从简单任务到复杂项目实现高效自动化编码。

OpenCode 是开源 AI 编码代理（类似 Claude Code / Cursor 的开源替代），支持终端、桌面、IDE 扩展。OMO 在其基础上添加编排层，让代理像“小团队”一样协作。

本手册基于 v3.2.1 版本（最新版，包含 Hephaestus 代理和多项 bug 修复），从零基础开始，逐步进阶到专家级用法。

## 1\. 引言：什么是 Oh-My-OpenCode？

### 核心优势

- **多代理协作** ：11 个专化代理并行工作（规划、研究、编码、审阅）。
- **两大模式** ： - **Ultrawork（ulw）** ：脑放空，全自动模式。输入目标，代理自主完成。 - **Prometheus + Atlas** ：精密规划模式。适合复杂/多会话任务。
- **电池包容** ：开箱即用，自动适配 Claude、GPT、Gemini 等模型。
- **隐私优先** ：不上传代码，支持本地/自控模型。
- **兼容性** ：完整支持 Claude Code 的钩子、技能、MCP。

### 适用人群

- 新手：快速实现功能。
- 专家：处理多仓库、复杂构建管道、重构、遗留系统。

### v3.2.1 新特性（相对于 v3.2.0）

- 修复后台代理并发槽泄漏。
- 支持 GitHub Copilot Gemini 模型预览。
- Hephaestus 代理（v3.2.0 引入）已稳定：目标导向的深度工作者，使用 GPT-5.2 Codex。

## 2\. 安装与设置（新手必读）

### 步骤 1：安装 OpenCode（前提）

```
curl -fsSL https://opencode.ai/install | bash
```

验证：

```
opencode --version  # 需 ≥ 1.0.150
```

### 步骤 2：安装 Oh-My-OpenCode

推荐方式（互动式，最简单）：

```
bunx oh-my-opencode install
```

或

```
npx oh-my-opencode install
```

安装程序会询问你的订阅情况（Claude Pro/Max、ChatGPT Plus、Gemini、GitHub Copilot 等），自动生成最佳配置。

### 步骤 3：认证提供商

运行：

```
opencode auth login
```

按提示选择：

- Anthropic（Claude）→ Claude Pro/Max OAuth
- Google（Gemini）→ Antigravity OAuth（支持多账号负载均衡）
- OpenAI / GitHub Copilot 等

### 步骤 4：验证安装

```
cat ~/.config/opencode/opencode.json  # 应包含 "oh-my-opencode"
opencode models  # 查看可用模型
```

### 卸载

1. 编辑 `~/.config/opencode/opencode.json` ，移除 "oh-my-opencode"
2. 删除配置文件：
```
rm -f ~/.config/opencode/oh-my-opencode.json
rm -f .opencode/oh-my-opencode.json
```

## 3\. 基本使用（从小白开始）

启动：

```
opencode
```

### Ultrawork 模式（最简单，全自动）

在提示中加入关键词 `ultrawork` 或 `ulw` ：

```
ulw 在我的 Next.js 项目中添加用户认证功能
```

代理会自动：

- 探索代码库
- 研究最佳实践
- 实施、测试、迭代直到完成

适合快速原型、修复 bug、添加小功能。

### Prometheus 精密模式（推荐用于复杂任务）

1. 按 **Tab** 键进入 Prometheus（规划者）模式
2. 描述任务，Prometheus 会提问澄清
3. 审阅生成的计划（位于 `.sisyphus/plans/*.md` ）
4. 输入 `/start-work` 启动 Atlas 执行

适合重构、多文件改动、跨会话项目。

### 手动调用专化代理

```
@oracle 审阅这个架构设计
@librarian 这个功能在开源项目中是怎么实现的？
@explore 搜索项目中所有 TODO
```

## 4\. 核心代理一览（理解团队分工）

| 代理名称 | 推荐模型 | 角色与特点 |
| --- | --- | --- |
| **Sisyphus** | anthropic/claude-opus-4-5 | 主编排者，Todo 驱动，全局协调并行执行 |
| **Hephaestus** | openai/gpt-5.2-codex | 深度工作者，目标导向，先探索后行动，精炼代码 |
| **oracle** | openai/gpt-5.2 | 架构、设计、代码审阅、调试（只读） |
| **librarian** | zai-coding-plan/glm-4.7 | 多仓库分析、文档检索、开源实现示例 |
| **explore** | anthropic/claude-haiku-4-5 | 快速代码库探索、模式匹配 |
| **multimodal-looker** | google/gemini-3-flash | 分析图片、PDF、设计图 |
| **Prometheus** | anthropic/claude-opus-4-5 | 规划者，通过访谈生成详细工作计划 |
| **Metis** | anthropic/claude-opus-4-5 | 计划前分析，识别隐藏意图和风险 |
| **Momus** | openai/gpt-5.2 | 计划审阅者，确保清晰、可验证 |

模型会根据你的订阅自动回退（原生 > Copilot > Zen > Z.ai）。

## 5\. 配置与自定义（中级）

配置文件： `~/.config/opencode/oh-my-opencode.json` （支持 JSONC 注释）

### 常用自定义示例

```
{
  "agents": {
    "oracle": { "model": "openai/gpt-5.2" },
    "explore": { "model": "anthropic/claude-haiku-4-5", "temperature": 0.3 },
    "multimodal-looker": { "disable": true}
},
"categories": {
    "visual-engineering": { "model": "google/gemini-3-pro-preview" },
    "ultrabrain": { "model": "openai/gpt-5.2-codex", "variant": "xhigh" }
},
"tmux": { "enabled": true},// 在 tmux 中可视化并行代理
"disabled_agents": ["multimodal-looker"]
}
```

### 类别（Categories）用途

用于 `delegate_task` 时指定领域模型，例如视觉任务自动用 Gemini。

### 后台任务并发

```
{
  "background_task": {
    "defaultConcurrency": 5,
    "providerConcurrency": { "anthropic": 3 }
  }
}
```

## 6\. 高级功能（专家级）

### 钩子（Hooks）

内置 25+ 钩子，可禁用：

```
{ "disabled_hooks": ["comment-checker"] }
```

重要钩子：

- todo-continuation-enforcer：强制完成 TODO
- ralph-loop：防止无限循环
- context-window-monitor：上下文管理

### 技能（Skills）

自定义技能，支持浏览器自动化（Playwright 或 agent-browser）。

### MCP（Model Context Protocol）

内置 websearch、context7、grep\_app，可禁用。

### LSP 支持

添加语言服务器（如 TypeScript）：

```
{
  "lsp": {
    "typescript-language-server": {
      "command": ["typescript-language-server", "--stdio"]
    }
  }
}
```

### 自定义技能与类别

创建复杂工作流，例如数据科学专用代理。

## 7\. 最佳实践与专家技巧

- **小任务** → 用 `ulw`
- **大任务** → 必用 Prometheus + Atlas
- **多会话项目** → 计划文件会自动保存，继续时直接 `/start-work`
- **模型选择** → Claude Opus 4.5 是 Sisyphus 最优；有 Gemini 时视觉任务自动优选
- **监控** → 启用 tmux 可视化并行代理执行
- **性能优化** → 限制并发、禁用不常用代理
- **安全** → 只读代理（oracle/librarian）不会修改代码
- **提示技巧** → 明确目标 + ulw = 最高效率

## 8\. 故障排除

- 配置不生效 → 检查 OpenCode 版本（>1.0.132），删除旧配置重装
- 模型不可用 → 运行 `opencode models` 检查，重新 auth
- 并发问题 → 查看后台任务日志
- 卡死 → 检查 tmux 或后台任务超时

## 9\. 更新

插件会自动检查更新，或手动：

```
bunx oh-my-opencode install
```

查看最新版本：https://github.com/code-yeongyu/oh-my-opencode/releases

恭喜！你已掌握 Oh-My-OpenCode 3.2.1。从现在起，让代理为你编码，享受真正的“Ultrawork”！如果有问题，欢迎查看 gitee.com/coxio/opencode-quickstart或私信。