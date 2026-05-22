HyperFrames 是一款以 HTML 为视频唯一可信源的视频渲染引擎，由 HeyGen 开发、Apache-2.0 开源，同时也是 NousResearch 开源 Hermes Agent 智能体框架的官方创意技能。
它可以直接作为技能安装到 Hermes Agent 中，让你的 AI 智能体直接具备专业级视频创作能力，你只用说一句话，AI 就能自动帮你写代码、做动画、渲染出成品视频，全程不用碰一行代码。

项目地址: https://github.com/heygen-com/hyperframes

### 安装
1. npx 安装
```bash
# 要求 Node.js >= 22，加 FFmpeg。
npx skills add heygen-com/hyperframes
# 装完之后直接在 agent 里描述需求即可
npx hyperframes init my-video
cd my-video
npx hyperframes preview      # 浏览器实时预览，热更新
npx hyperframes render       # 渲染成 MP4
```
2. hermes agent里安装
```
登录 hermes 交换命令，可以让大模型给你安装
● hermes skills install hyperframes
────────────────────────────────────────
  ┊ 🐍 preparing execute_code…
  ╎ 🐍 exec      from hermes_tools import terminal  12.0s
──────────────────────────────────────────────────────────────────────────────────╮
    所有 HyperFrames 相关的 skills 已经安装好了，都可以正常加载使用。以下是完整的 skill 列表：

    1. **hyperframes** - 核心 skill：创建 HTML 视频合成、动画标题卡、字幕、音频反应动画、场景过渡等
    2. **hyperframes-cli** - CLI 开发循环：init、lint、inspect、preview、render、doctor 等命令
    3. **hyperframes-media** - 媒体预处理：TTS 语音合成 (Kokoro)、音频转录 (Whisper)、背景移除 (u2net)
    4. **hyperframes-registry** - 注册表：安装和连接可复用的 blocks 和 components
    5. **remotion-to-hyperframes** - 从 Remotion (React) 迁移到 HyperFrames (HTML+GSAP)
    6. **website-to-hyperframes** - 抓取网站并生成 HyperFrames 视频
    7. **waapi** - Web Animations API 适配器模式

    这些 skills 通过符号链接从 `~/.agents/skills/` 映射到 `~/.hermes/skills/`，每个都可以正常通过 `skill_view` 加载。

    之前 `hermes skills install` 失败是因为 GitHub API 限速（未认证每小时 60 次请求），但 skills 本身已经就位，不影响使用。
```

3. hermes交互里， 初始化看看hyperframes的demo效果， 会自动创建一段宣传hyperframes的视频。类似ppt展示的效果。
![[自媒体工具/hyperframes工具/assets/hyperframes/e1d582fb26ab71f50f4184676a05dbf1_MD5.png]]