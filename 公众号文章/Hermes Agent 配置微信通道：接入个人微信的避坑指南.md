---
title: "Hermes Agent 配置微信通道：接入个人微信的避坑指南"
source: "https://blog.csdn.net/bhl120/article/details/160044493"
author:
  - "[[bhl120]]"
published: 2026-04-11
created: 2026-04-22
description: "文章浏览阅读3.3k次，点赞9次，收藏12次。用了一段时间，微信通道的稳定性比预期好。但说实话，个人微信的 API 限制是硬伤，比如不能主动发消息给陌生人、容易被封等。如果你要做正经的客服/助手场景，企业微信（WeCom）方案会更合适。个人微信接入更多是尝鲜或者个人使用。另外提醒：微信对第三方接入一直不太友好，且用且珍惜。遵守平台规则，不要发垃圾消息。Hermes Agent 安装部署：60秒入门这个可成长的AI助手Hermes Agent 配置飞书通道：让机器人跑在 Lark 上配置微信通道（本文）_hermes agent 微信"
tags:
  - "clippings"
---
前两篇写了 Hermes Agent 的安装和飞书配置，这篇说说怎么接微信。

说实话，接入个人微信这件事，踩坑是大概率事件。微信本身封闭，第三方接入方案多多少少都有点灰色地带。Hermes Agent 用的是腾讯官方的 iLink Bot API，理论上是最稳的方案，但配置过程还是有一些需要注意的地方。

### 重要说明

这是 **个人微信** 接入，不是企业微信。个人微信的 API 限制比较多，稳定性也不能跟企业微信比。如果你要的是企业微信，请看 WeCom 适配器（官方有单独的文档）。

### 前提条件

- Hermes Agent 已安装
- 一个个人微信账号
- Python 包： `aiohttp` 和 `cryptography`
```bash
pip install aiohttp cryptography
pip install qrcode  # 可选，用于终端显示二维码
bash
```

### 工作原理

Hermes 通过腾讯的 **iLink Bot API** 接入微信，用的是长轮询（long-polling）方式，不需要公网地址或 WebSocket。流程大概是：

1. 手机微信扫码授权
2. Gateway 通过 iLink API 轮询拉取消息
3. 处理后通过同一 API 发送回复

### Step 1: 运行设置向导

最简单的方式是用交互式向导：

```bash
hermes gateway setup
bash1
```

选「Weixin」，向导会：

1. 请求 iLink Bot API 的二维码
2. 在终端显示二维码（或提供 URL）
3. 用微信手机版扫码
4. 在手机上确认登录
5. 自动保存凭证到 `~/.hermes/weixin/accounts/`

扫码成功后显示：

```
微信连接成功，account_id=your-account-id
1
```

### Step 2: 手动配置（如需）

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

### Step 3: 启动 Gateway

```bash
hermes gateway
bash1
```

适配器会恢复保存的凭证，连接 iLink API，开始长轮询拉取消息。

### 访问控制策略

#### 私聊策略

```bash
WEIXIN_DM_POLICY=open        # 任何人可以私聊（默认）
WEIXIN_DM_POLICY=allowlist   # 只有白名单用户可以私聊
WEIXIN_DM_POLICY=disabled    # 完全忽略私聊
bash123
```

#### 群策略

```bash
WEIXIN_GROUP_POLICY=open        # 所有群都响应
WEIXIN_GROUP_POLICY=allowlist   # 只响应白名单群
WEIXIN_GROUP_POLICY=disabled    # 忽略所有群消息（默认）
bash123
```

⚠️ 默认禁用群策略，因为个人微信可能加了很多群，避免刷屏。

### 核心功能

接入后支持：

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

### 媒体处理

#### 接收

- **图片** ：下载后 AES 解密，缓存为 JPEG
- **视频** ：下载后 AES 解密，缓存为 MP4
- **文件** ：下载后 AES 解密，保留原始文件名
- **语音** ：如果有文本 transcription 就用文本，否则缓存 SILK 格式音频

#### 发送

通过加密 CDN 上传：

1. 生成本地 AES-128 密钥
2. 用密钥加密文件
3. 上传到微信 CDN
4. 发送加密后的媒体引用

这些都是自动处理的，不需要手动配置。

### 消息格式化

微信个人版对 Markdown 支持有限，适配器会自动转换：

- `# 标题` → `【标题】`
- `## 标题` → `**标题**`
- 表格 → 转换为键值列表
- 代码块 → 保持原样（微信渲染还行）

### 常见问题

**Q: Weixin startup failed: aiohttp and cryptography are required**  
A: 安装依赖： `pip install aiohttp cryptography`

**Q: Weixin startup failed: WEIXIN\_TOKEN is required**  
A: 重新运行 `hermes gateway setup` 完成扫码登录，或手动设置 `WEIXIN_TOKEN`

**Q: Another local Hermes gateway is already using this Weixin token**  
A: 先停掉另一个 Gateway 实例，一个 token 只能被一个实例使用

**Q: Session expired (errcode=-14)**  
A: 登录会话过期，重新运行 `hermes gateway setup` 扫码

**Q: QR code expired during setup**  
A: 二维码 会自动刷新最多 3 次。如果持续过期，检查网络连接

**Q: Bot doesn’t respond to DMs**  
A: 检查 `WEIXIN_DM_POLICY` ，如果是 `allowlist` ，发送者必须在 `WEIXIN_ALLOWED_USERS` 里

**Q: Bot ignores group messages**  
A: 默认群策略是 `disabled` 。设置 `WEIXIN_GROUP_POLICY=open` 或 `allowlist`

**Q: Media download/upload fails**  
A: 确保 `cryptography` 已安装。检查网络能否访问 `novac2c.cdn.weixin.qq.com`

**Q: 终端二维码不显示**  
A: 安装 qrcode ： `pip install qrcode` 。或者使用向导输出的 URL 在浏览器扫码

### 技术细节

#### 长轮询机制

- 每次请求 35 秒超时
- 服务器hold住请求直到有消息或超时
- 入站消息通过 asyncio 并发分发
- 重试策略：瞬时错误等2秒重试，连续错误等30秒

#### Token 持久化

context\_token 会保存到 `~/.hermes/weixin/accounts/.context-tokens.json` ，确保重启后对话连贯。

#### 去重

5 分钟滑动窗口去重，避免网络抖动导致重复处理。

### 写在最后

用了一段时间，微信通道的稳定性比预期好。但说实话，个人微信的 API 限制是硬伤，比如不能主动发消息给陌生人、容易被封等。

如果你要做正经的客服/助手场景，企业微信（WeCom）方案会更合适。个人微信接入更多是尝鲜或者个人使用。

另外提醒：微信对第三方接入一直不太友好，且用且珍惜。遵守平台规则，不要发垃圾消息。

系列文章：

- Hermes Agent 安装部署：60秒入门这个可成长的AI助手
- Hermes Agent 配置飞书通道：让机器人跑在 Lark 上
- 配置微信通道（本文）