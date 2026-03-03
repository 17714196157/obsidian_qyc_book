
### 安装
##### 安装前准备
```
# 1. 先安装 Node.js（如未安装）
# 访问 https://nodejs.org/ 下载 LTS 版本并安装
# 2. 验证 Node.js 安装
node -v
npm -v
```
 管理员权限打开cmd
 
##### 各个工具的安装命令
1.  claude-code安装： npm install -g @anthropic-ai/claude-code
2.  opencode安装：npm install -g opencode-ai@latest


配置介绍一个非常好用的开源工具：**CCSwitch**（地址：[cc-switchv3.11.1 · farion1231/cc-switch · GitHub](https://github.com/farion1231/cc-switch/releases/tag/v3.11.1)）


opencode的 oh-my-opencode 插件安装：
```
# Ubuntu/Debian 用 Snap 装的 Bun 可能会报错，建议用官方脚本重装：
curl -fsSL https://bun.sh/install | bash

# 什么订阅都没有（用免费 Provider）
bunx oh-my-opencode install --no-tui --claude=no --chatgpt=no --gemini=no

# 检查插件是否加载 ,确保 `plugin` 数组里包含 `oh-my-opencode`
cat ~/.config/opencode/opencode.json | grep -A5 '"plugin"'

```
