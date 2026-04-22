Claude Code 的安装配置、免登录方案以及更换底层模型为 DeepSeek 的相关信息。
##### 一、安装 Claude Code
##### 安装前准备
```
# 0.安装 nvm  管理nodejs版本
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install 20 # 安装 Node.js 20 的最新版本
nvm alias default 20  # 指定具体版本
node -v # 此时应显示 v20.x.x 等高于你之前版本的号码

# 1. 先安装 Node.js（如未安装）
# 访问 https://nodejs.org/ 下载 LTS 版本并安装
# 2. 验证 Node.js 安装
node -v
npm -v
```
 管理员权限打开cmd
 
方式 1：官方脚本安装（推荐）
```bash
# macOS / Linux
curl -fsSL https://claude.ai/install.sh | bash

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```
方式 2：NPM 安装
```bash
npm install -g @anthropic-ai/claude-code
```
验证安装：
```bash
claude --version
```
---
## 二、免登录配置+配置 DeepSeek 作为底层模型
Claude Code 默认需要登录 Anthropic 账号，通过修改配置文件可以跳过登录：
### 1. 创建/编辑配置文件

**macOS/Linux:**
```bash
vim ~/.claude.json
```
**Windows:**
```bash
# 文件路径
C:\Users\你的用户名\.claude.json
```
在文件末尾添加：
```json
{
  "hasCompletedOnboarding": true
}
```

配置 DeepSeek 作为底层模型
DeepSeek 提供与 **Anthropic API 兼容的接口**（需要参考大模型服务商官网文档，变量名有区别，url也不一样），只需设置环境变量即可切换 ：
```
# 创建目录
mkdir C:\Users\你的用户名\.claude
```
写入内容：
```json
{ 
  "hasCompletedOnboarding": true,

  "env": {
  "ANTHROPIC_AUTH_TOKEN": "sk-b5e02d8f907b42f98044391e97f854ab",
    "ANTHROPIC_BASE_URL": "https://coding.dashscope.aliyuncs.com/apps/anthropic",
        "ANTHROPIC_MODEL": "deepseek-v3",
        "ANTHROPIC_SMALL_FAST_MODEL": "deepseek-v3",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v3",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v3",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v3",
        "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-v3"
  },

}
```
验证当前模型：
```
/status
```
或在对话中询问："你现在使用的是什么模型？"
