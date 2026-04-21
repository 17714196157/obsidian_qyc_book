官网文档 https://hermes-agent.nousresearch.com/docs/getting-started
安装：
```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc

hermes model # Choose your LLM provider and model  
hermes tools # Configure which tools are enabled  
hermes setup # Or configure everything at once
hermes  # 启动交互应用界面

hermes chat -q "测试"

```
**配置文件路径： Hermes Agent 的主配置文件 `~/.hermes/config.yaml`**

启动UI管理界面 
```bash
hermes dashboard --host 0.0.0.0 --port 1111 --insecure
```
![[Hermes-AgentUI界面.png]]

###  hermes 链接个人微信

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
bash123456789101112
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

