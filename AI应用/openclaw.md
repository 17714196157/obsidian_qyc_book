- **官方文档**：https://docs.openclaw.ai
- **GitHub 仓库**：https://github.com/openclaw/openclaw
- **中文官网**：https://openclaws.io/zh/install
- **图形界面工具**（Windows）：https://github.com/miaoxworld/openclaw-manager/releases

## 🔧 安装

- **Node.js** ≥ 22（推荐使用 nvm 管理）
方式一：一键脚本安装（推荐）
**macOS / Linux：**
```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```
**Windows（PowerShell）：**
```powershell
# 需要先安装 WSL2
wsl --install
# 重启后进入 WSL，然后执行 Linux 安装命令
```
国内网络加速版（解决 GitHub 访问慢的问题）：
```bash
# 1. 先配置 GitHub 镜像
git config --global url."https://hub.fastgit.xyz/".insteadOf "https://github.com/"
git config --global url."https://hub.fastgit.xyz/".insteadOf "git@github.com:"

# 2. 切换国内 npm 镜像
npm config set registry https://registry.npmmirror.com

# 3. 运行安装脚本
curl -fsSL https://openclaw.ai/install.sh | bash
```
方式二：npm 全局安装
```bash
# 安装 Node.js 22（如未安装）
nvm install 22 && nvm use 22
# 全局安装 OpenClaw
npm config set registry https://mirrors.cloud.tencent.com/npm/ && npm install -g openclaw@latest

openclaw --version

which openclaw 2>/dev/null || whereis openclaw
>>> /root/.nvm/versions/node/v22.22.1/bin/openclaw
sudo ln -sf /root/.nvm/versions/node/v22.22.1/bin/openclaw /usr/local/bin/openclaw

```
方式三：Docker 安装（**最安全，推荐服务器使用**）
```bash
# 1. 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw
# 2. 运行 Docker 安装脚本
./docker-setup.sh
# 3. 启动服务
docker compose up -d
# 4. 查看状态
docker compose ps

直接 docker run 方式：
docker run -d \
  --name openclaw \
  -v ~/.openclaw:/root/.openclaw \
  -p 18789:18789 \
  ghcr.io/openclaw/openclaw:latest \
  gateway --port 18789
```
```
```
## ⚙️ 初始化配置
安装完成后，必须运行配置向导：
openclaw onboard  # 开始配置
```
◇  QuickStart ─────────────────────────╮
│  Gateway port: 18789                 │
│  Gateway bind: Loopback (127.0.0.1)  │
│  Gateway auth: Token (default)       │
│  Tailscale exposure: Off             │
│  Direct to chat channels.            │
├──────────────────────────────────────╯
◇  Model/auth provider
│  Custom Provider
│
◇  API Base URL
│  https://dashscope.aliyuncs.com/compatible-mode/v1
│
◇  How do you want to provide this API key?
│  Paste API key now
│
◇  API Key (leave blank if not required)
│  sk-b5e02d8f907b42f98044391e97f854ab
│
◇  Endpoint compatibility
│  OpenAI-compatible
│
◇  Model ID
│  deepseek-v3
│
◇  Verification successful.
│
◇  Endpoint ID
│  custom-dashscope-aliyuncs-com
│
◇  Model alias (optional)
│  aliyuncs
Configured custom provider: custom-dashscope-aliyuncs-com/deepseek-v3
◇  How do you want to hatch your bot?
│  Open the Web UI
◇  Dashboard ready 
│  Dashboard link (with token):                                                    │
│  http://127.0.0.1:18789/#token=2f9a44d1197b8193e6efc0adcb2d4c7b6d524c9dac3a414b  │
│  Opened in your browser. Keep that tab to control OpenClaw.                    
                                                       
```
[[openclaw界面.png|Open: file-20260306014103033.png]]
![[openclaw界面.png]]
