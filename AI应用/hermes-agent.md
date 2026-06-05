官网文档 https://hermes-agent.nousresearch.com/docs/getting-started
**安装：**
```bash
# 安装 nvm  管理nodejs版本
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install 22 # 安装 Node.js 22 的最新版本
nvm alias default 22  # 指定具体版本
node -v # 此时应显示 v22.x.x 等高于你之前版本的号码

# 安装 hermes-agent
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes update # 升级版本
hermes model # Choose your LLM provider and model  
hermes tools # Configure which tools are enabled  
hermes setup # Or configure everything at once
hermes  # 启动交互应用界面

hermes chat -q "测试"

```
 **一键清理脚本:**
```bash
 #!/bin/bash
echo "=== 卸载 Hermes Agent ==="
pip uninstall hermes-agent -y 2>/dev/null

echo "=== 卸载 Hermes Studio ==="
sudo apt remove --purge hermes-studio -y 2>/dev/null

echo "=== 清理所有配置和缓存 ==="
# Hermes 主配置目录
rm -rf ~/.hermes
# Studio 配置（Electron 应用）
rm -rf ~/.config/hermes-studio
rm -rf ~/.config/Hermes\ Studio
# 缓存文件
rm -rf ~/.cache/hermes*
rm -rf ~/.local/share/hermes*
# 日志文件
rm -rf ~/.local/state/hermes*
rm -f /root/.local/bin/hermes

echo "=== 完成 ==="
which hermes 2>/dev/null || echo "hermes 已完全移除"
# 检查是否还有残留
npm list -g | grep -i hermes
```

**配置文件路径： Hermes Agent 的主配置文件 `~/.hermes/config.yaml`**

启动UI管理界面 
```bash
hermes dashboard --host 0.0.0.0 --port 1111 --insecure
```
![[Hermes-AgentUI界面.png]]

**windows下安装方法，参考如下**
![[AI应用/assets/hermes-agent/6629f88adc69cb9222cb68700cd563c3_MD5.png]]

###  一）hermes 链接个人微信

```bash 
pip install aiohttp cryptography
pip install qrcode
```
Hermes 通过腾讯的 **iLink Bot API** 接入微信，用的是长轮询（long-polling）方式，不需要公网地址或 WebSocket。流程大概是：
1. 手机微信扫码授权
2. Gateway 通过 iLink API 轮询拉取消息
3. 处理后通过同一 API 发送回复

- Step 1: 运行设置向导
最简单的方式是用交互式向导：
```bash
hermes gateway setup
```
![[951a64846f159cab71e6d9e531d2cfbc_MD5.png]]
选「Weixin」，向导会：
1. 请求 iLink Bot API 的二维码
2. 在终端显示二维码（或提供 URL）
3. 用微信手机版扫码
4. 在手机上确认登录
5. 自动保存凭证到 `~/.hermes/weixin/accounts/`
- Step 2: 手动配置（如需）
向导会自动保存凭证，但如果需要手动设置，编辑 `~/.hermes/.env` ：
```bash
WEIXIN_ACCOUNT_ID=your-account-id
WEIXIN_TOKEN=your-bot-token  # 通常向导已自动保存

# 可选：访问控制
WEIXIN_DM_POLICY=open              # 私聊策略：open/allowlist/disabled
WEIXIN_ALLOWED_USERS=user_id_1   # 白名单用户
WEIXIN_GROUP_POLICY=disabled      # 群策略：默认禁用，避免在所有群里响应
WEIXIN_GROUP_ALLOWED_USERS=group_id_1

# 可选：首页频道（收到 cron 通知的地方）
WEIXIN_HOME_CHANNEL=chat_id
WEIXIN_HOME_CHANNEL_NAME=Home
```
- Step 3: 启动 Gateway
```bash
hermes gateway
```
适配器会恢复保存的凭证，连接 iLink API，开始长轮询拉取消息。
**访问控制策略**
 私聊策略
```bash
WEIXIN_DM_POLICY=open        # 任何人可以私聊（默认）
WEIXIN_DM_POLICY=allowlist   # 只有白名单用户可以私聊
WEIXIN_DM_POLICY=disabled    # 完全忽略私聊
bash123
```
群策略
```bash
WEIXIN_GROUP_POLICY=open        # 所有群都响应
WEIXIN_GROUP_POLICY=allowlist   # 只响应白名单群
WEIXIN_GROUP_POLICY=disabled    # 忽略所有群消息（默认）
```
核心功能，接入后支持：
- ✅ 长轮询传输（不需要公网地址）
- ✅ 二维码扫码登录
- ✅ 私聊和群聊
- ✅ 图片、视频、文件、语音消息
- ✅ AES-128-ECB 加密的 CDN 传输
- ✅ 消息上下文记忆（跨重启持久化）
- ✅ Markdown 格式化（会自动转换）
- ✅ 智能消息分片（长消息自动拆分）
- ✅ 打字状态提示
- ✅ SSRF 保护
---
### 二） Hermes监控台
**项目地址：** github.com/EKKOLearnAI/hermes-web-ui
```
npm install -g hermes-web-ui 

root@maizi:/home/qyc/gitee/hermes-agent# hermes-web-ui stop
  ✓ hermes-web-ui stopped (PID: 4128987)
  
root@maizi:/home/qyc/gitee/hermes-agent#  hermes-web-ui start --host 0.0.0.0

  ⏳ Starting hermes-web-ui (PID: 4137609, port: 8648)...
  ✓ hermes-web-ui started
    http://localhost:8648/#/?token=f56904d3b78a9bb085ba075b2827b1c29bf06d41bdddba5b758fd4e68845f5e7
    Log: /root/.hermes-web-ui/server.log
```


![[hermes-web-ui界面.png]]

---
### 三） Hermes斜杠命令

在 Hermes 的对话界面里，输入 `/` 会弹出一个自动补全菜单，列出所有可用的命令。命令不分大小写，`/HELP` 和 `/help` 效果一样。

装了的技能也会自动变成斜杠命令，比如装了 `plan` 技能后，输 `/plan` 就能调用。

#### 1. 基础操作类

##### 1.1 `/new` 和 `/reset` —— 清屏重来
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
##### 1.2 `/title` —— 给会话起个名字
**什么时候用**：这个会话你想留着以后恢复，起个名字好找。
```
❯ /title 用户登录功能开发
```
之后恢复：
```
hermes -c "用户登录功能开发"
```
##### 1.3 `/save` —— 手动保存会话
**什么时候用**：做了重要操作，想确保不会丢。
```
❯ /save
```
默认 Hermes 退出时自动保存，但关键时刻手动存一下更安心。
##### 1.4 `/quit` —— 退出
```
❯ /quit
```
退出前会自动保存会话，并显示恢复命令。
#### 2. 模型切换类
##### 2.1 `/model` —— 查看和切换模型
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

#### 3. 工具和信息类
##### 3.1 `/tools` —— 查看当前可用的工具
**什么时候用**：想知道 Hermes 现在能干什么，或者排查某个功能为什么没反应。
```
❯ /tools
```
会列出所有当前启用的工具，比如 `terminal`、`web_search`、`read_file` 等。如果你发现 Hermes 不会搜索网页，先 `/tools` 看看 `web_search` 在不在列表里。
##### 3.2 `/memory` —— 查看记忆内容
**什么时候用**：好奇 Hermes 记住了你什么，或者想确认某条信息有没有被记进去。
```
❯ /memory
```
显示当前 `MEMORY.md` 和 `USER.md` 的内容，带占用比例。
##### 3.3 `/compress` —— 压缩上下文
**什么时候用**：聊了很久，状态栏变橙色/红色了，或者 Hermes 开始忘事。
```
❯ /compress
```
Hermes 会把中间部分的历史对话总结成摘要，腾出空间。经常用于处理复杂任务聊到一半的时候。
##### 3.4 `/usage` —— 查看 Token 和花费统计
**什么时候用**：想知道这回合花了多少 token、总共花了多少钱。
```
❯ /usage
```
显示详细的输入/输出 token 数、预估花费。
##### 3.5 `/insights` —— 使用统计
**什么时候用**：想看看最近用了多少、花在哪些模型上。
```
❯ /insights --days 7        # 最近7天
❯ /insights --days 30       # 最近30天
```
#### 4. 人格和风格类
##### 4.1 `/personality` —— 切换人格
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

##### 4.2 `/reasoning` —— 调整推理深度
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
#### 5. 技能类
##### 5.1 `/skills` —— 管理技能
```
❯ /skills                           # 列出已安装技能
❯ /skills browse                    # 浏览技能中心
❯ /skills search kubernetes         # 搜索技能
❯ /skills install openai/skills/k8s # 安装技能
```
##### 5.2 `/技能名` —— 调用技能
装了技能后，技能名自动变成斜杠命令：
```
❯ /plan 帮我设计一个用户认证系统的方案
❯ /github-pr-workflow 给这个功能创建个PR
```
#### 6. 会话管理类
##### 6.1 `/checkpoint` 和 `/restore` —— 检查点
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
##### 6.2 `/busy` —— 控制打断行为
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

#### 7. 显示控制类
##### 7.1 `/verbose` —— 切换输出详细程度
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

##### 7.2 `/skin` —— 切换界面皮肤
```
❯ /skin                     # 查看当前皮肤
❯ /skin 皮肤名               # 切换皮肤
```
#### 8. 语音类
##### 8.1 `/voice` —— 语音模式
```
❯ /voice on                 # 开启语音输入
❯ /voice off                # 关闭
❯ /voice tts                # 开关语音播报回复
```
开了 `/voice on` 后，按 `Ctrl+B` 录音，再按一下停止。Hermes 自动转文字处理。
#### 9. 后台任务类
##### 9.1 `/background` —— 后台执行
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
#### 10. 其他实用命令

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
#### 11. 自定义快捷命令
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

#### 12. 作者日常 Workflow
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

#### 13. 命令速查表

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

### 三）Hermes"工具集"
```
hermes tools
```
这是个交互式界面，用方向键选择平台（CLI、Telegram、Discord等），空格键勾选/取消工具集，回车保存。

如果你只想用某些工具，启动时直接指定：
`hermes chat --toolsets "web,terminal,file"`
这样就不会加载其他工具，省token，也减少误操作风险。

| 工具集          | 包含的能力     | 备注                                                                                                                                                                                                                                                                                                                                   |
| :----------- | :-------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `web`        | 网页搜索、内容提取 | **web搜索的后端选择**：<br>默认用Firecrawl，需要 `FIRECRAWL_API_KEY`。国内用户推荐配 `TAVILY_API_KEY` 或 `EXA_API_KEY`，都挺好用。<br>如果你有自己的SearXNG搜索实例，也可以配 `SEARXNG_URL`，完全免费。                                                                                                                                                                                 |
| `terminal`   | 执行终端命令    | 让Hermes能在你的系统上执行命令。这是最强大的工具，也是最有风险的。<br>`❯ 当前目录下哪个文件夹占空间最大？`<br>Hermes会自动运行 `du -sh * \| sort -hr`，然后把结果告诉你。<br>`❯ 帮我创建一个备份脚本，每天自动备份Documents文件夹`<br>它会写脚本、测试、告诉你怎么用。<br>**安全提示**：terminal工具默认会审批危险命令（rm、dd之类）。你可以在配置里调整审批模式：<br>`approvals:  mode: manual    # 手动审批（默认）  # mode: smart   # 智能审批，低风险自动过  # mode: off     # 关闭审批，危险！` |
| `file`       | 读写文件、搜索文件 | 读写文件、搜索文件内容。<br>`❯ 帮我读一下config.yaml的内容`   <br>`❯ 在readme.md末尾加一段安装说明  ` <br>`❯ 搜索项目里所有包含"TODO"的文件`                                                                                                                                                                                                                                   |
| `browser`    | 浏览器自动化    | 让Hermes控制浏览器，访问网页、填表单、截图。<br>`❯ 打开 https://news.ycombinator.com，告诉我前三条新闻的标题`<br>`❯ 帮我截图这个网页`                                                                                                                                                                                                                                         |
| `vision`     | 图像识别分析    |                                                                                                                                                                                                                                                                                                                                      |
| `memory`     | 记忆管理      |                                                                                                                                                                                                                                                                                                                                      |
| `cronjob`    | 定时任务      |                                                                                                                                                                                                                                                                                                                                      |
| `skills`     | 技能系统      |                                                                                                                                                                                                                                                                                                                                      |
| `todo`       | 待办事项      |                                                                                                                                                                                                                                                                                                                                      |
| `delegation` | 子代理委派     |                                                                                                                                                                                                                                                                                                                                      |