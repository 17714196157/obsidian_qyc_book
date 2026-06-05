# TrendRadar Docker 安装指南

> 使用 Docker 容器方式部署 TrendRadar，接入阿里百炼（DashScope）大模型，实现舆情监控与热点分析。

---

## 一、环境准备

**确认 Docker 环境**
```bash
docker --version
docker-compose --version
```
需要 Docker ≥ 20.10 且已安装 `docker-compose` 插件或独立版本。

**获取项目**
使用浅克隆加快下载速度：
```bash
cd /tmp
git clone --depth 1 https://github.com/sansan0/TrendRadar.git
```

---
## 二、项目结构概览

克隆完成后，关键文件分布如下：

```
TrendRadar/
├── config/                 # 配置文件目录（核心）
│   ├── config.yaml         # 主配置文件
│   ├── frequency_words.txt # 关键词词表
│   ├── ai_interests.txt    # AI 兴趣描述
│   ├── timeline.yaml       # 调度时间线
│   └── ai_analysis_prompt.txt  # AI 分析提示词
├── docker/                 # Docker 相关文件
│   ├── docker-compose.yml  # 编排文件（注意：不在项目根目录！）
│   ├── .env                # 环境变量模板
│   ├── Dockerfile          # 构建用（使用官方镜像无需关心）
│   └── entrypoint.sh       # 容器启动脚本
├── trendradar/             # 源代码
└── output/                 # 输出目录（运行时生成）
```

> **⚠️ 注意点 1：** 项目的 `docker-compose.yml` 位于 `docker/` 子目录下，不在项目根目录。直接使用会报 "配置文件缺失" 错误（volume 路径 `../config` 相对位置不对）。

---

## 三、部署步骤

### 3.1 创建部署目录

```bash
mkdir -p /home/qyc/TrendRadar
cd /home/qyc/TrendRadar

# 复制配置文件和 docker-compose.yml 到工作目录
cp -r /tmp/TrendRadar/config ./
cp /tmp/TrendRadar/docker/docker-compose.yml .
cp /tmp/TrendRadar/docker/.env .
```

> **为什么这样做？** 把配置和编排文件放到同一目录，volume 路径用 `./config` 即可，避免相对路径混乱。

### 3.2 修改 volume 路径

编辑 `docker-compose.yml`，将 volume 路径从 `../config` 改为 `./config`：

```yaml
# 修改前（docker/ 子目录下的原始配置）
volumes:
  - ../config:/app/config:ro
  - ../output:/app/output

# 修改后（放到项目根目录后的配置）
volumes:
  - ./config:/app/config:ro
  - ./output:/app/output
```

> **⚠️ 注意点 2：** 这是最常见的坑。如果不改这个路径，容器会报 `❌ 配置文件缺失` 并反复重启。

### 3.3 配置阿里百炼 API

TrendRadar 使用 **LiteLLM** 作为 AI 适配层，这意味着所有模型都遵循 `provider/model_name` 格式。

**修改 `.env` 文件**
```bash
# ============================================
# AI 配置（ai_analysis 和 ai_translation 共享模型配置）
# ============================================
# 是否启用 AI 分析 (true/false)
AI_ANALYSIS_ENABLED=true
# AI API Key（必填，启用 AI 功能时需要）
AI_API_KEY=sk-d98a7434af1f4641921b8af02e175499
# AI 模型名称（LiteLLM 格式: provider/model_name）
# 示例: deepseek/deepseek-chat, openai/gpt-4o, gemini/gemini-2.5-flash
AI_MODEL=openai/qwen3.6-plus
# 自定义 API 端点（可选，大多数情况留空）
AI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

修改 `config/config.yaml` 文件
```yaml
# 8. AI 模型配置（ai_analysis / ai_translation / ai_filter 共用）
ai:
  # LiteLLM 模型格式: 提供商/模型名
  # 示例:
  #   - deepseek/deepseek-v4-flash (DeepSeek，便宜够用，推荐)
  #   - deepseek/deepseek-v4-pro
  #   - openai/gpt-4o (OpenAI)
  #   - gemini/gemini-2.5-flash (Google Gemini)
  #   - anthropic/claude-sonnet-4-20250514 (Anthropic)
  #   - ollama/llama3 (本地 Ollama)
  # 完整列表: https://docs.litellm.ai/docs/providers
  # 如果你对于看英文文档比较头疼，那么可以点击页面右下角的 【Ask AI】 ,用中文询问怎么配置
  model: "openai/qwen3.6-plus"
  api_key: "sk-d98a7434af1f4641921b8af02e175499"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

> **⚠️ 注意点 3：阿里百炼的模型命名**
> 
> 阿里百炼（DashScope）的兼容接口是 OpenAI 格式的，所以：
> - **必须**在模型名前加 `openai/` 前缀，例如 `openai/qwen3.6-plus`
> - **不能**直接写 `qwen3.6-plus`，LiteLLM 无法识别
> - API Base 必须填 `https://dashscope.aliyuncs.com/compatible-mode/v1`
> 可选模型：`qwen-turbo`、`qwen-plus`、`qwen-max`、`qwen3.6-plus` 等。

### 3.4 配置外部访问端口（可选）

默认 Web 端口是 8080。如需改为其他端口（如 8383），编辑 `docker-compose.yml`：

```yaml
ports:
  - "0.0.0.0:8383:8080"
  #   ↑ 外部端口   ↑ 容器内部端口
```

- `0.0.0.0` 表示允许任何 IP 访问（局域网/远程可访问）
- `127.0.0.1` 表示仅本机可访问

> **⚠️ 注意点 4：内外端口分离**
> 
> 外部端口和内部端口可以不同。容器内的 Web 服务始终监听 `.env` 中 `WEBSERVER_PORT` 指定的端口（默认 8080），docker-compose 的 `ports` 负责做端口映射。
> 
> 正确的做法是：`.env` 中保持 `WEBSERVER_PORT=8080`，`ports` 中写 `"0.0.0.0:8383:8080"`。

### 3.5 启动容器
```bash
# 拉取镜像
docker-compose pull trendradar

# 启动服务
docker-compose up -d trendradar
```

> **⚠️ 注意点 5：MCP 服务可选**
> 
> `docker-compose.yml` 中定义了两个服务：`trendradar`（主服务）和 `trendradar-mcp`（MCP Server）。
> - MCP 镜像较大，国内网络可能拉取超时
> - 如果不需要 MCP 功能（自然语言对话分析），只启动主服务即可：`docker-compose up -d trendradar`
> - 需要 MCP 时再单独启动：`docker-compose up -d trendradar-mcp`

---

## 四、验证与排障

### 4.1 检查容器状态

```bash
docker ps --filter name=trendradar
docker logs trendradar --tail 30
```

正常日志应显示：
```
配置文件加载成功: /app/config/config.yaml
TrendRadar v6.9.0 配置加载完成
监控平台数量: 11
[AI] 模型: openai/qwen3.6-plus
```

### 4.2 Web 服务器的启动时机

> **⚠️ 注意点 6：Web 服务不是立即启动的**
> 
> 查看 `entrypoint.sh` 可知，容器启动流程为：
> 1. 校验配置文件
> 2. 立即执行一次数据抓取（如果 `IMMEDIATE_RUN=true`）
> 3. 调用 AI 分析（如果启用）
> 4. AI 翻译（如果启用）
> 5. **最后**才启动 Web 服务器
> 
> 这意味着从 `docker-compose up` 到 Web 服务可用，可能需要 **1~3 分钟**（取决于 AI 分析速度）。期间 `curl` 会返回连接失败，这是正常的，耐心等待即可。

验证：
```bash
curl -s -o /dev/null -w "HTTP %{http_code}" http://<服务器IP>:8383/index.html
# 正常返回 HTTP 200
```

### 4.3 常见错误速查

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `❌ 配置文件缺失` 反复重启 | volume 路径不对 | 改 `../config` 为 `./config` |
| AI 分析报 `model not found` | 模型名缺少 provider 前缀 | 改为 `openai/qwen3.6-plus` |
| Web 服务连接失败 | 还在执行 AI 分析，Web 尚未启动 | 等 1~3 分钟后重试 |
| `docker compose` 命令找不到 | 未安装 compose 插件 | 使用 `docker-compose`（短横线版） |
| 无法远程访问 | 端口绑定了 `127.0.0.1` | 改为 `0.0.0.0` |
| 防火墙拦截 | 系统防火墙未放行 | `iptables/firewalld/ufw` 放行端口 |

---

## 五、常用运维命令

```bash
# 查看实时日志
docker logs -f trendradar

# 查看配置是否生效（检查 AI 配置）
docker logs trendradar 2>&1 | grep -i "AI\|model\|api"

# 修改配置后重启
cd /home/qyc/TrendRadar
docker-compose up -d --force-recreate trendradar

# 停止服务
docker-compose down

# 查看输出文件（HTML 报告等）
ls -la output/html/
ls -la output/html/latest/

# 清理旧数据
docker-compose down -v  # 会删除容器和 volume（不会删除挂载的 config/output 目录）
```

---

## 六、核心配置文件说明

### config.yaml 关键段

| 配置段 | 作用 | 说明 |
|--------|------|------|
| `app.timezone` | 时区 | 默认 `Asia/Shanghai` |
| `platforms.sources` | 监控平台 | 可增删热搜源 |
| `rss.feeds` | RSS 订阅 | 添加自定义 RSS 源 |
| `filter.method` | 筛选方式 | `keyword`（关键词）或 `ai`（AI 分类） |
| `report.mode` | 报告模式 | `current`（当前榜）/ `daily`（当日汇总）/ `incremental`（增量） |
| `notification.channels` | 通知渠道 | 飞书/钉钉/Telegram/邮件等 |
| `schedule.enabled` | 调度系统 | `true` 启用按时间表推送 |
| `ai` | AI 模型配置 | API Key、模型、端点 |
| `ai_analysis.enabled` | AI 分析 | 是否生成分析简报 |
| `ai_translation.enabled` | AI 翻译 | 是否翻译外文标题 |

### .env 环境变量（docker-compose 注入）

| 变量 | 作用 |
|------|------|
| `WEBSERVER_PORT` | Web 服务端口 |
| `AI_API_KEY` | AI API 密钥 |
| `AI_MODEL` | 模型名称（`openai/模型名`） |
| `AI_API_BASE` | 自定义 API 端点 |
| `RUN_MODE` | `cron`（定时）或 `once`（单次） |
| `CRON_SCHEDULE` | cron 表达式，默认 `*/30 * * * *` |

### 数据源与选题策略
##### 1. 新增 LLM/AI 垂直数据源
已在 `config/config.yaml` 的 `rss.feeds` 中增加以下源：
*   **OpenAI Blog** (官方更新)
*   **TechCrunch AI** (行业新闻)
*   **Hugging Face Blog** (开源社区)
*   **ArXiv CS.AI** (最新论文)
##### 2. 全量推送模式（自媒体选题专用）
为了不错过任何热点，已将 `frequency_words.txt` 修改为全量匹配。

**修改 `config/frequency_words.txt`：**
```text
[GLOBAL_FILTER]
震惊
广告

[WORD_GROUPS]
# 正则表达式 /.+/ 匹配所有非空标题，实现全量推送
/.+/ => 全部热点
```


---

## 七、进阶：配置通知推送

TrendRadar 支持多种通知渠道。以飞书为例：
##### 7.1配置飞书，将信息同步给飞书
在飞书中**添加自定义机器人**并获取 Webhook 地址的步骤如下，
1. **打开飞书群聊**，点击群聊右上角的 **「设置」**（或「更多」按钮）
2. 在右侧设置面板中找到 **「群机器人」** → 点击 **「添加机器人」**
3. 在弹出的窗口中选择 **「自定义机器人」**（Custom Bot）
获取 Webhook 地址
![[自媒体工具/选题工具/assets/UP数据收集/9aa8fdaae8449d36541c3bd905cc42bf_MD5.png]]
![[自媒体工具/选题工具/assets/UP数据收集/1d87b9ffa6259da87297da20589829d6_MD5.png]]
测试给飞书群发消息：
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"msg_type":"text","content":{"text":"Hello from Hermes"}}' \
  https://open.feishu.cn/open-apis/bot/v2/hook/67fc539f-d9f0-4067-981e-61dadea6d227
```
![[自媒体工具/选题工具/assets/UP数据收集/301ea71ace23469de718883fc7c1f8d5_MD5.png]]

##### 7.2 在 config.yaml 中配置
```yaml
#    GitHub 部署请将 webhook 填入 GitHub Secrets，不要写在这里
# 📌 多账号：分号(;)分隔，如 "url1;url2;url3"
#    配对项（如 Telegram token 和 chat_id）数量必须一致

notification:
  enabled: true #是否启用通知功能（true=启用, false=关闭）—— 总开关
 # 开启 schedule 后此项仍为总开关：false=永远不推送，true=由调度控制
  # 推送渠道配置
  channels:
    feishu:
      webhook_url: "https://open.feishu.cn/open-apis/bot/v2/hook/67fc539f-d9f0-4067-981e-61dadea6d227"


```

---

## 八、文件清单（部署后）

```
/home/qyc/TrendRadar/
├── config/               # 配置文件（只读挂载到容器）
│   ├── config.yaml       # 主配置
│   ├── frequency_words.txt
│   ├── ai_interests.txt
│   └── ...
├── output/               # 输出数据（读写挂载到容器）
│   ├── html/             # HTML 报告
│   ├── news/             # 新闻数据库
│   └── rss/              # RSS 数据
├── docker-compose.yml    # 编排文件
├── .env                  # 环境变量
└── 安装指南.md            # 本文档
```

---

*本指南基于 TrendRadar v6.9.0 编写，配置格式可能随版本更新变化。*
