---
title: "GPT-Image2 完全指南：一期讲透主流玩法（附我的生图 Skill）"
url: "https://www.bilibili.com/video/BV1vi9UBhEKq?spm_id_from=333.788.videopod.sections&vd_source=d0a50f3d250eed1f7d1546f70041c66b"
bvid: "BV1vi9UBhEKq"
cid: "37938988735"
author: "code秘密花园"
upload_date: "2026-04-29"
subtitle_lang: "中文"
created: "2026-05-19"
tags:
  - clippings
  - bilibili
  - gpt-image-2
  - image-generation
  - skill
  - claude-code
  - prompt-engineering
---

<iframe src="https://player.bilibili.com/player.html?aid=116486895108819&bvid=BV1vi9UBhEKq&cid=37938988735&page=1&autoplay=0" scrolling="no" border="0" frameborder="no" framespacing="0" allow="fullscreen; picture-in-picture" allowfullscreen="true" style="height:100%;width:100%; aspect-ratio: 16 / 9;"> </iframe>

## 概述

全面介绍 GPT-Image-2.0 的核心能力、实测高质量玩法，以及如何通过 Skill 稳定复现效果。

> [!tip] 核心观点
> GPT-Image-2 在 Are.na AI 生图排行榜拿到 **1512 高分**，比第二名 NanoBanana 高 200+ 分，OpenAI 官方都表示从未有任何模型能拉开这么大的差距。在大部分场景下是对 NanoBanana 的**碾压级超越**。

---

## 相关资源

| 资源 | 地址 |
|------|------|
| GPT-Image-2 案例网站（含提示词） | https://gpt-image2.mmh1.top/ |
| 开源生图 Skill 仓库 | https://github.com/ConardLi/garden-skills/ |

---

## GPT-Image-2 三大核心优势

| 优势 | 说明 |
|------|------|
| **1. 文字渲染** | 多行文本、不同字体嵌套、复杂画面中的文字都能正确渲染。海报封面、PPT 配图、信息图场景手拿把掐 |
| **2. 指令遵循** | 可指定主体位置、背景渐变、文字竖排、材质风格等具体需求，准确响应，可做专业产品设计图 |
| **3. 照片级真实感** | 光影、材质、人物接近真实商业摄影效果，消除过去 AI 生图的过度饱和、光照不自然、材质过于光滑等问题 |

---

## 六大典型玩法

### 1. UI 截图伪造

> [!warning] 伦理提醒
> 此能力展示了模型的逼真程度，"有图有真相"正在成为过去式。请负责任地使用。

模型擅长生成**看起来像真实截图**的 UI 界面效果：

| 类型 | 效果 |
|------|------|
| 假微信聊天界面 | 以假乱真 |
| 假小红书截图 | 细节精确 |
| 假直播间截图 | 界面元素完整 |
| 假 Twitter 截图 | 布局精准 |

**提示词特点**：使用标准 JSON 结构，直接替换字段内容即可精准复刻效果。

### 2. 产品视觉与营销海报

直接在提示词中指定：
- 品牌名称、Slogan
- 配色方案
- 人物站位
- 核心产品描述

> 最终效果可能比初级设计师还要好。

**案例**：化妆品海报、Vision Pro 核心部件拆解图、开源项目介绍海报。

### 3. 信息图与数据可视化

得益于文字渲染能力的大幅提升，以下类型均可**稳定生成**：

- 手绘风格信息图
- 多步骤教程图
- 时间演进图（含大量信息）
- 图文混合高密度多模块图（每个区块中文标签清晰）

### 4. 学术绘图

> [!note] 意外惊喜
> 之前用 NanoBanana 尝试科研配图总是差点意思。GPT-Image-2 应该用了大量专业论文配图做训练，最终效果跟正经论文投稿里的 Figure 一样。

### 5. 角色与漫画

- **多格漫画分镜**
- **角色设定表**
- **角色关系图**

> 这些以前需要找画师的活，现在模型能直接交付。可用于 AI 短剧前期的角色设计和分镜生成。简单工作流即可将小说转换为漫画效果。

### 6. 技术架构图

| 图类型 | 说明 |
|--------|------|
| 系统架构图 | 软件/系统组件关系 |
| 流程图 | 业务流程、算法流程 |
| 时序图 | 交互时序 |
| ER 图 | 实体关系 |
| 状态图 | 状态机转换 |
| 思维导图 | 知识梳理 |
| 网络拓扑图 | 网络架构 |

选择需要的风格，使用对应模板创建即可。

### 其他玩法

风格化头像、人物视觉效果、地图生成、图标设计、游戏资源等。完整几百个案例可在案例网站查看。

---

## 生图 Skill 设计

> [!abstract] 什么是 Skill？
> Skill 是一套给 Agent 看的**工作手册**，可放入 Claude Code、Cursor、Codex 等 Agent 环境。Agent 会按 Skill 定义的流程，通过**渐进式披露**原则加载对应资源并完成任务。

### 工作流程

```
用户需求
  │
  ▼
┌─────────────────────────┐
│ 1. 检查本地环境          │  → 有没有生图 API / 图像工具？
├─────────────────────────┤
│ 2. 分析视觉类型          │  → 海报？论文配图？信息图？
├─────────────────────────┤
│ 3. 匹配结构化模板        │  → 找到对应模板
├─────────────────────────┤
│ 4. 补充信息              │  → 按模板要求向用户提问
├─────────────────────────┤
│ 5. 渲染高质量提示词      │  → 生成结构化 prompt
├─────────────────────────┤
│ 6. 调用工具/API 出图     │  → 根据环境选择执行方式
└─────────────────────────┘
```

### 三种运行模式

| 模式 | 条件 | 说明 |
|------|------|------|
| **A. 全自动模式** | 有 GPT-Image-2 API Key（OpenAI 开放平台 / OpenRouter） | 配置环境变量后，一句话完成从选模板到出图的全流程 |
| **B. 委托宿主模式** | 宿主 Agent 自带图像生成工具（如 Codex） | Skill 走完选模板和提示词渲染流程，最终提示词交给宿主自己的生图工具执行，无需单独配置 API Key |
| **C. 纯顾问模式** | 无生图能力、无 API Key | Skill 走完模板选择流程，输出高质量提示词文件，手动复制到 ChatGPT 等生图平台使用 |

> [!tip] 模式 C 的价值
> 虽然多了一步手动操作，但提示词质量比直接手写高很多。

---

## 安装与使用

### 安装 Skill

1. 克隆仓库到本地：`git clone https://github.com/ConardLi/garden-skills/`
2. 复制 `skills/` 目录下对应技能到指定位置
3. 根据使用的 Agent 创建对应目录：
   - Claude Code → `.claude/skills/`
   - Cursor → `.cursor/skills/`
   - Codex → `.codex/skills/`

### 使用演示

#### 模式 C（顾问模式）— 最简单

无需任何配置，直接对话：
1. Agent 自动识别为模式 C
2. 匹配提示词模板
3. 根据模板要求向用户提问（或使用默认值）
4. 生成 `garden-gpt-image2/` 目录下的提示词文件
5. 复制提示词到 ChatGPT 等平台出图

#### 模式 A（全自动模式）— 直接出图

1. 创建 `.env` 文件，配置环境变量：
   ```
   ENABLE_GARDEN_IMAGINE=true
   OPENAI_BASE_URL=your_openai_base_url
   OPENAI_API_KEY=your_api_key
   ```
2. 重启 Agent
3. Agent 识别为模式 A，完成提问后直接生成图片到目录

#### 模式 B（委托宿主模式）

在 Codex 等自带生图工具的 Agent 中：
1. 使用快捷命令找到已添加的 Skill
2. Agent 自动识别为模式 B（无环境变量但宿主有生图工具）
3. 完成模板匹配 → 生成提示词 → 调用宿主工具直接出图

### 自定义输出目录

默认输出到 `garden-gpt-image2/` 目录，可在输入任务时直接指定输出位置。

---

## 总结

> [!success] 快速上手
> 对 GPT-Image-2 生图感兴趣，做两件事：
> 1. **案例网站翻一翻** → 找到感兴趣的方向，直接复制提示词试试效果
> 2. **配置 Garden Skills** → 如果你在用 Codex / Claude Code / Cursor，拉取仓库配置 Skill，以后说一句话即可直接出图

模板和案例会持续更新，可在 GitHub 仓库 Star 支持，有问题直接提 Issue。
