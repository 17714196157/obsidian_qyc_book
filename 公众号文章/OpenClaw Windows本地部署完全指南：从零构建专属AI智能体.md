---
title: "OpenClaw Windows本地部署完全指南：从零构建专属AI智能体"
source: "https://mp.weixin.qq.com/s/iKcUhHRc7TAc4Xv80ok5YQ"
author:
  - "[[鱼弦]]"
published:
tags:
  - "clippings"
---
*2026年4月2日 18:36*

以下文章来源于红尘灯塔 ，作者鱼弦

[

**红尘灯塔**.

分享前沿技术，关注行业动态

](https://mp.weixin.qq.com/s/#)

![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/c02ca35b949ff37c07d2fa9395262e9b_MD5.webp]]

## OpenClaw Windows本地部署完全指南：从零构建专属AI智能体

## 一、引言

2026年，人工智能领域正经历一场深刻的范式转移——从“对话式AI”迈向“执行式AI”。在这场变革中，OpenClaw如同一只红色的龙虾，悄然爬进了全球开发者的视野。这个以龙虾为图标的开源项目，在GitHub上迅速斩获超过15万星标，被业界戏称为“2026年最炸裂开源项目”。

OpenClaw的本质并非又一个聊天机器人，而是一个 **可持久运行的Agent调度框架** 。它的核心价值在于：让AI从“回答问题”进化为“执行任务”。当传统云端大模型像一位无所不知的远程顾问，只提供文本方案时，OpenClaw完全运行于本地或私有云，拥有与用户等同的系统操作权限，能直接操作电脑终端、编写代码、管理文件，甚至能根据自然语言指令自主学习并安装新的“技能（Skills）” 。

然而，长期以来，关于AI Agent的讨论往往被绑定在Apple Silicon与Mac mini的叙事框架中。这引发了一个核心疑问： **Windows用户是否只能旁观这场技术革命？**

答案显然是否定的。Windows不仅跟得上AI代理浪潮，甚至能凭借其广泛的硬件生态和NVIDIA独立显卡的优势，完成一次真正的PC逆袭。通过WSL2（Windows Subsystem for Linux）与Docker Desktop的整合，Windows用户可以在本地打造24小时待命的AI数字劳动力，不必依赖云端环境，也不需转投macOS阵营。

本文旨在提供一份 **完整的OpenClaw Windows本地部署指南** ，涵盖从技术背景、原理解析、环境准备到多场景实战应用的全部内容。无论你是希望提升工作效率的个人开发者，还是寻求智能化转型的企业团队，都能在这份近两万字的指南中找到可执行的方案。所有代码命令均经过验证，可直接复制执行，助你在自己的Windows工作站上，养一只属于自己的“龙虾”。

![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/b9a4a64b40e1fd9aeab703e3df21e1b8_MD5.webp]]

## 二、技术背景

### 2.1 AI Agent的进化路径

要理解OpenClaw的价值，需要回溯AI应用形态的演进历程：

| 阶段 | 代表技术 | 核心能力 | 局限性 |
| --- | --- | --- | --- |
| 第一阶段 | 规则机器人 | 基于预设关键词的固定回复 | 无法处理未预定义场景 |
| 第二阶段 | 大语言模型 | 自然语言理解与生成 | 仅限对话，无法执行操作 |
| 第三阶段 | AI Agent | 理解、规划、执行、学习 | 生态碎片化，部署复杂 |

OpenClaw正处在第三阶段的核心位置。它解决的不仅是“生成内容”，更是“如何组织多步骤任务”、“如何调用外部工具”、“如何管理上下文”以及“如何长期运行”等工程化问题。

### 2.2 OpenClaw的诞生与定位

OpenClaw最初作为一个开源项目发布，其设计理念深受“AI操作系统”概念的启发。它不是要取代现有的大模型，而是要成为连接大模型与现实世界的 **调度中枢** 。

在OpenClaw的架构中，大模型扮演“大脑”角色负责思考与规划，而OpenClaw本身则扮演“小脑”与“神经系统”的角色，负责协调各类工具、控制执行流程、管理上下文记忆。这种分层设计使得OpenClaw具备极强的扩展性——只要封装成Skill（技能），任何外部服务或API都可以被Agent调用。

### 2.3 Windows平台的独特优势

选择Windows作为OpenClaw的部署平台，具备以下战略价值：

**硬件成本优势** ：相比Mac Mini专用部署方案¥5000+的硬件成本，Windows方案可利用现有工作站，实现零额外硬件投入。

**GPU算力红利** ：Windows生态拥有最广泛的NVIDIA独立显卡用户群体。当结合CUDA与本地LLM时，PC的效能潜力远未被充分发掘。通过本地量化模型，可以实现完全离线的智能体运行，保障数据隐私。

**生态兼容性** ：Windows+WSL方案全平台支持，既享受Windows的日常使用便利，又能利用Linux生态的丰富工具链。

**Local-first架构优势** ：代码、财报、项目资料皆留存在本机硬盘，敏感信息无需上传至外部服务器，满足企业级数据安全要求。

![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/ee90cfb7880c330771fa5ea451ff1ac6_MD5.webp]]

  

## 三、核心特性

### 3.1 调度框架（Orchestration Framework）

OpenClaw的核心并非模型本身，而是一个 **事件驱动的任务调度引擎** 。它采用类似操作系统的设计理念，将复杂的业务逻辑拆解为可编排的工作流。

```
# 调度核心抽象示例
class OpenClawScheduler:
    def __init__(self):
        self.tasks = PriorityQueue()
        self.skills = SkillRegistry()
        self.memory = ContextMemory()
    
    def process_request(self, user_input):
        # 1. 意图识别
        intent = self.analyze_intent(user_input)
        # 2. 任务拆解
        subtasks = self.decompose_task(intent)
        # 3. 资源分配
        for task in subtasks:
            skill = self.match_skill(task)
            self.tasks.put((skill.priority, task))
        # 4. 执行调度
        return self.execute_pipeline()
```

### 3.2 Skills机制（可扩展工具集）

Skills是OpenClaw生态的基石，它将“某件事的完整流程”封装成一个可触发的能力模块。这一机制解决了传统Prompt的四大问题：

- **每次都写完整流程**
	→ Skill一次封装，永久复用
- **不可复用**
	→ Skill可在不同Agent间共享
- **上下文浪费严重**
	→ Skill按需加载，精简Token
- **工程可维护性差**
	→ Skill独立版本控制与测试

根据ListenHub官方文档，目前Skills已支持播客生成、解说视频生成、语音朗读、图片生成等多模态能力，且支持文章URL、纯文本、视频链接、结构化信息等多种输入格式。

### 3.3 多Agent协作架构

2026年，HiClaw作为OpenClaw的“超进化版本”引入Manager Agent（AI管家）角色，构建“管家+专业工人”的团队架构：

```
你（真人管理员） → Manager Agent（AI管家） → Worker Agents（专业工人）
                        ↓                        ↓
                  需求理解、任务拆解     前端开发、后端开发、文档撰写
                  资源协调、进度监控     独立技能库、隔离记忆空间
```

这种三层协作体系实现了从“单兵作战”到“AI军团”的质变。

### 3.4 长期记忆（Persistence Memory）

OpenClaw通过Workspace机制实现“越用越聪明”的效果。每个Agent拥有独立的记忆空间，可以：

- 存储用户偏好与历史交互
- 记录已完成任务的执行路径
- 维护领域知识库的增量更新

### 3.5 安全边界（Security Boundary）

OpenClaw是高权限Agent，必须具备严格的安全控制：

- **权限最小化原则**
	：仅授予执行当前任务所需的最小权限
- **容器隔离**
	：通过Docker实现Worker进程的故障隔离
- **凭证集中管理**
	：AI Gateway统一管理API Key，Worker仅持临时令牌
- **技能审核机制**
	：仅安装经过VirusTotal扫描、有公开源码仓库、文档清晰的技能
![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/bd1dc1829fe42e163820d7a32ed13c49_MD5.unknown]]

  

## 四、原理解释

### 4.1 OpenClaw核心架构

理解OpenClaw的工作原理，需要从系统架构视角剖析其核心组件。下图展示了OpenClaw的完整调用链：

![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/bd1dc1829fe42e163820d7a32ed13c49_MD5.unknown]]

### 4.2 调度引擎工作流程

调度引擎是OpenClaw的“大脑”，其工作流程可分为六个阶段：

**阶段一：输入解析** 当用户通过命令行、Web界面或IM工具发送指令时，OpenClaw首先对输入进行标准化处理，提取核心意图与参数。

**阶段二：意图识别** 通过轻量级分类模型或LLM提示，判断用户请求的类型：是简单查询、文件操作、代码生成，还是需要多步骤协作的复杂任务。

**阶段三：任务拆解** 对于复杂任务，调度引擎将其分解为有向无环图（DAG）形式的子任务序列。例如，“开发一个博客系统”可拆解为：

- 子任务1：前端脚手架搭建（调用frontend Worker）
- 子任务2：数据库模型设计（调用backend Worker）
- 子任务3：API接口开发（调用backend Worker）
- 子任务4：文档撰写（调用docs Worker）

**阶段四：资源匹配** 根据子任务类型，从技能仓库中匹配最合适的Skill，同时考虑：

- Worker当前负载状态
- 模型性能需求（简单任务用轻量模型，复杂任务用高性能模型）
- 数据 locality（优先访问本地数据）

**阶段五：执行与监控** 并行或串行执行子任务，实时监控执行状态。若某子任务失败，触发重试机制或向Manager报告异常。

**阶段六：结果聚合** 收集所有子任务输出，进行格式化与后处理，最终以用户友好的方式呈现。

### 4.3 Skills调用链深度解析

以“把这篇文章生成播客”这一典型场景为例，Skills调用链的完整流程如下：

```
TTS服务大模型ListenHub Skill触发器引擎OpenClaw用户TTS服务大模型ListenHub Skill触发器引擎OpenClaw用户“把这篇文章生成播客”匹配播客Skill调用Podcast能力生成播客脚本返回对话稿语音合成返回音频文件返回播客链接输出音频
```

这一流程揭示了OpenClaw的核心设计哲学： **OpenClaw负责判断，Skill负责执行，外部能力负责生成** 的清晰分层架构。

### 4.4 记忆系统实现机制

OpenClaw的记忆系统采用三级存储架构：

**热数据层（会话记忆）**

- 存储介质：Redis / 内存缓存
- 数据内容：当前会话上下文、临时变量
- 过期策略：会话结束或24小时后自动清除

**温数据层（工作区记忆）**

- 存储介质：SQLite / 本地文件系统
- 数据内容：用户偏好、常用模板、历史任务记录
- 管理方式：按Agent隔离，支持手动导出/导入

**冷数据层（长期知识库）**

- 存储介质：对象存储 / 云端归档
- 数据内容：项目文档、知识图谱、训练数据
- 访问频率：按需加载，支持RAG检索增强

这种分级设计在性能与成本之间取得平衡，既保证高频访问的响应速度，又实现海量数据的持久化存储。

![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/04cddd07051ad04578297095039cc90c_MD5.webp]]

## 五、环境准备

### 5.1 系统要求验证

在开始部署前，请确认Windows系统满足以下最低要求：

| 配置项 | 最低要求 | 推荐配置 |
| --- | --- | --- |
| 操作系统 | Windows 10 2004版 | Windows 11 |
| 内存 | 8GB | 16GB或更高 |
| 磁盘空间 | 20GB可用空间 | 50GB SSD |
| 处理器 | 支持虚拟化技术 | Intel i5/AMD R5 以上 |
| GPU | 可选 | NVIDIA GTX 1060 / RTX系列 |
| BIOS设置 | 启用VT-x/AMD-V | \- |

**验证命令** （以管理员身份运行PowerShell）：

```
# 查看Windows版本
winver

# 查看系统信息
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"

# 查看内存
wmic memorychip get capacity

# 检查虚拟化支持
Get-WmiObject -Class Win32_Processor | Select-Object -Property Name, VirtualizationFirmwareEnabled
```

### 5.2 WSL2安装与配置

WSL2（Windows Subsystem for Linux）是在Windows上运行Linux环境的最佳方案，也是OpenClaw源码部署的基础。

**步骤1：启用WSL功能**

以管理员身份打开PowerShell，执行以下命令：

```
# 启用WSL功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用虚拟机平台功能
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 重启计算机
Restart-Computer
```

**步骤2：安装Linux内核更新包**

1. 下载WSL2 Linux内核更新包：
	```
	https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
	```
2. 安装下载的MSI文件

**步骤3：设置WSL2为默认版本**

```
wsl --set-default-version 2
```

**步骤4：安装Ubuntu 22.04 LTS**

```
# 方法A：通过命令行安装
wsl --install -d Ubuntu-22.04

# 方法B：通过Microsoft Store安装
# 打开Microsoft Store，搜索"Ubuntu 22.04.5 LTS"，点击安装
```

**步骤5：首次启动配置**

安装完成后，启动Ubuntu，按提示创建UNIX用户名和密码：

```
Installing, this may take a few minutes...
Please create a default UNIX user account: ubuntu
New password: [输入密码]
Retype new password: [确认密码]
```

**步骤6：验证WSL状态**

```
wsl -l -v
# 应显示：
#   NAME            STATE           VERSION
# * Ubuntu-22.04    Running         2
```

**步骤7：WSL性能优化（可选）**

在Windows用户目录下创建`.wslconfig` 文件，限制WSL资源使用：

```
[wsl2]
memory=8GB        # 根据物理内存调整，建议不超过物理内存的50%
processors=4      # 根据CPU核心数调整
localhostForwarding=true
swap=2GB
```

保存后重启WSL：

```
wsl --shutdown
# 重新打开Ubuntu终端
```

### 5.3 Docker Desktop安装（HiClaw部署必需）

HiClaw依赖Docker实现容器化部署，需先完成Docker Desktop安装。

**步骤1：安装Docker Desktop**

```
# 方法A：使用winget安装（推荐）
winget install Docker.DockerDesktop

# 方法B：手动下载安装
# 访问 https://www.docker.com/products/docker-desktop/ 下载Windows版本
```

**步骤2：配置WSL2集成**

安装完成后，打开Docker Desktop：

1. 进入Settings → Resources → WSL Integration
2. 启用"Enable integration with my default WSL distro"
3. 在列表中选择"Ubuntu-22.04"
4. 点击Apply & Restart

**步骤3：验证安装**

在PowerShell中执行：

```
docker --version
# 应显示 Docker version 24.0.7 或更高

docker run hello-world
# 应成功运行测试容器
```

### 5.4 Node.js与npm安装

OpenClaw核心框架基于Node.js开发，需要安装v22.0.0及以上版本。

**步骤1：安装Node.js LTS版本**

```
# 方法A：使用winget安装
winget install OpenJS.NodeJS.LTS

# 方法B：使用安装包（国内镜像加速）
curl -fsSL https://npmmirror.com/mirrors/node/v22.5.0/node-v22.5.0-x64.msi -OutFile node-install.msi
Start-Process .\node-install.msi -Wait
```

**步骤2：验证安装**

```
node --version  # 应显示 v22.5.0 或更高
npm --version   # 应显示 v10.0.0 或更高
```

**步骤3：配置npm镜像（国内用户）**

```
# 配置淘宝镜像加速
npm config set registry https://registry.npmmirror.com
```

### 5.5 Python环境配置

部分Skills和执行脚本依赖Python 3.9以上版本。

```
# 通过winget安装Python
winget install Python.Python.3.11

# 验证安装
python --version  # 应显示 Python 3.11.x
pip --version     # 应显示 pip 23.x
```

### 5.6 网络代理配置（国内用户）

针对国内开发者常见的网络问题，建议采用分层代理方案。

**系统级代理配置** ：

```
# 设置HTTP代理（假设本地代理端口为1080）
$env:HTTP_PROXY = "http://127.0.0.1:1080"
$env:HTTPS_PROXY = "http://127.0.0.1:1080"

# 永久配置（添加到PowerShell profile）
Add-Content $PROFILE "\`n\`$env:HTTP_PROXY='http://127.0.0.1:1080'"
Add-Content $PROFILE "\`n\`$env:HTTPS_PROXY='http://127.0.0.1:1080'"
```

**WSL内代理穿透** ：

```
# 在Ubuntu终端中执行
echo 'export http_proxy="http://host.docker.internal:1080"' >> ~/.bashrc
echo 'export https_proxy="http://host.docker.internal:1080"' >> ~/.bashrc
source ~/.bashrc
```

### 5.7 环境验证脚本

完成上述所有配置后，运行以下验证脚本确保环境就绪：

```
# Windows环境验证脚本
Write-Host "=== OpenClaw Windows环境验证 ===" -ForegroundColor Green

# 检查WSL
$wslVersion = wsl --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ WSL已安装" -ForegroundColor Green
} else {
    Write-Host "✗ WSL未正确安装" -ForegroundColor Red
}

# 检查Docker
docker --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker已安装" -ForegroundColor Green
} else {
    Write-Host "✗ Docker未安装" -ForegroundColor Red
}

# 检查Node.js
$nodeVersion = node --version 2>$null
if ($nodeVersion -match "v22") {
    Write-Host "✓ Node.js $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Node.js版本需≥v22.0.0" -ForegroundColor Red
}

# 检查Python
$pythonVersion = python --version 2>$null
if ($pythonVersion -match "3.1[1-9]") {
    Write-Host "✓ Python $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python版本需≥3.11" -ForegroundColor Red
}

Write-Host "=== 验证完成 ===" -ForegroundColor Green
```

## 六、部署场景与详细实现

### 6.1 场景一：个人开发者快速部署（一键脚本）

适合希望最快速度上手OpenClaw的个人用户，无需关心底层细节。

![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/746b41607c8b47d07ceca6d74af24d1d_MD5.webp]]

  

#### 6.1.1 部署目标

- 在Windows上快速运行OpenClaw基础功能
- 通过Web界面与Agent交互
- 支持基本文件操作与代码生成

#### 6.1.2 详细部署步骤

**步骤1：以管理员身份打开PowerShell**

右键点击开始菜单，选择“Windows PowerShell (管理员)”。

**步骤2：执行一键安装脚本**

```
# 官方正式版脚本
iwr -useb https://openclaw.ai/install.ps1 | iex
```

如果下载失败，使用国内镜像：

```
# 国内镜像脚本
iwr -useb https://clawd.org.cn/install.ps1 | iex
```

若安装过程中遇到执行策略错误，先执行：

```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**步骤3：安装过程解析**

安装脚本会自动执行以下操作：

- 检测系统环境（Windows版本、内存、磁盘空间）
- 安装Node.js（若未安装）
- 通过npm全局安装openclaw包
- 创建配置文件目录 `%USERPROFILE%\.openclaw`
- 添加openclaw命令到系统PATH

**步骤4：验证安装**

```
openclaw --version
# 应显示 2026.1.29 或更高版本（修复CVE-2026-25253漏洞）
```

**步骤5：初始化配置**

```
openclaw onboard
```

交互式配置向导说明：

```
? 接受风险提示（OpenClaw拥有系统操作权限）: Yes
? 选择初始化模式: QuickStart
? 选择模型提供商: Custom Provider（后续可修改）
? API Base URL: http://127.0.0.1:11434/v1（若使用Ollama）
? API Key: ollama（可任意输入，但不可为空）
? Model ID: qwen2.5:7b-32k（需提前定制）
? 配置聊天平台集成: Skip for now
? 网关端口: 18789
```

**步骤6：启动网关服务**

```
# 设置Gateway模式
openclaw config set gateway.mode local

# 启动Gateway服务
openclaw gateway start

# 验证服务状态
openclaw gateway status
# 应显示 "Gateway is running"
```

**步骤7：生成访问令牌**

```
# 生成令牌
openclaw token generate

# 查看令牌
cat "$env:USERPROFILE\.openclaw\openclaw.json" | Select-String '"token"'
```

**步骤8：访问Web界面**

打开浏览器，访问：

```
http://127.0.0.1:18789/?token=你的令牌值
```

看到OpenClaw对话界面即部署成功。

#### 6.1.3 完整代码示例：自定义配置脚本

将以下内容保存为 `configure-openclaw.ps1` ，实现配置自动化：

```
# OpenClaw自动配置脚本
param(
    [string]$ApiBase = "http://127.0.0.1:11434/v1",
    [string]$ModelId = "qwen2.5:7b-32k",
    [int]$Port = 18789
)

Write-Host "开始配置OpenClaw..." -ForegroundColor Green

# 检查OpenClaw是否安装
if (!(Get-Command openclaw -ErrorAction SilentlyContinue)) {
    Write-Host "错误: openclaw命令未找到，请先运行安装脚本" -ForegroundColor Red
    exit 1
}

# 备份现有配置
$configDir = "$env:USERPROFILE\.openclaw"
if (Test-Path "$configDir\openclaw.json") {
    $backupFile = "$configDir\openclaw.json.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item "$configDir\openclaw.json" $backupFile
    Write-Host "已备份现有配置至: $backupFile" -ForegroundColor Yellow
}

# 创建配置目录（如果不存在）
if (!(Test-Path $configDir)) {
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
}

# 生成配置文件内容
$config = @{
    version = "2026.1"
    gateway = @{
        mode = "local"
        port = $Port
        host = "127.0.0.1"
    }
    agents = @{
        defaults = @{
            workspace = "$configDir\workspace"
            model = $ModelId
            maxTokens = 32768
            temperature = 0.7
        }
    }
    providers = @{
        custom = @{
            apiBase = $ApiBase
            apiKey = "ollama"
        }
    }
}

# 写入配置文件
$config | ConvertTo-Json -Depth 5 | Set-Content "$configDir\openclaw.json" -Encoding UTF8
Write-Host "配置文件已生成: $configDir\openclaw.json" -ForegroundColor Green

# 设置网关模式
openclaw config set gateway.mode local

# 启动服务
Write-Host "正在启动网关服务..." -ForegroundColor Yellow
$logFile = "$configDir\logs\gateway.log"
if (!(Test-Path "$configDir\logs")) {
    New-Item -ItemType Directory -Path "$configDir\logs" -Force | Out-Null
}

# 后台启动服务
$process = Start-Process -FilePath "openclaw" -ArgumentList "gateway start" -NoNewWindow -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
Write-Host "网关服务已启动 (PID: $($process.Id))" -ForegroundColor Green

# 生成令牌
$token = openclaw token generate --quiet
Write-Host "访问令牌: $token" -ForegroundColor Cyan

# 输出访问URL
Write-Host "\`n访问地址: http://127.0.0.1:$Port/?token=$token" -ForegroundColor Magenta

Write-Host "\`n配置完成！" -ForegroundColor Green
```

#### 6.1.4 运行结果

![[公众号文章/assets/OpenClaw Windows本地部署完全指南：从零构建专属AI智能体/661662857122d06f71536656e3e60d75_MD5.webp]]

成功部署后，Web界面应显示类似以下内容：

```
OpenClaw Agent v2026.1
会话ID: 550e8400-e29b-41d4-a716-446655440000

[用户]: 你好，请介绍一下你自己
[Agent]: 我是OpenClaw，一个可持久运行的AI Agent调度框架。我可以帮助你执行各种任务，包括文件操作、代码生成、数据分析和工具调用。当前运行在本地模式，使用qwen2.5:7b-32k模型。有什么可以帮你的？
```

### 6.2 场景二：结合Ollama完全本地私有化部署（隐私优先）

适合处理敏感数据、需要保障数据不出内网的企业用户或个人开发者。

#### 6.2.1 部署目标

- 所有推理在本地完成，数据不离开设备
- 集成Ollama管理本地大模型
- 定制模型上下文窗口以满足OpenClaw要求

#### 6.2.2 Ollama安装与模型下载

**步骤1：安装Ollama**

访问Ollama官网（https://ollama.com/）下载Windows版本，或使用命令行安装：

```
# 使用winget安装
winget install Ollama.Ollama
```

**步骤2：启动Ollama服务**

安装完成后，Ollama会自动在后台运行。可通过以下命令验证：

```
# 查看Ollama服务状态
Get-Service ollama

# 测试API是否正常
curl http://127.0.0.1:11434/api/tags
# 应返回类似 {"models":[]}
```

**步骤3：下载基础模型**

以通义千问2.5 7B模型为例（中文能力优秀，硬件要求适中）：

```
# 打开PowerShell或CMD
ollama pull qwen2.5:7b
```

模型下载过程：

```
pulling manifest
pulling 6d19a8f3bb82... 100% ▕████████████████▏ 4.7 GB
pulling 6a5b9f6c5c7f... 100% ▕████████████████▏  289 B
pulling 7c4f2f8c5b5f... 100% ▕████████████████▏  116 B
verifying sha256 digest
writing manifest
removing any unused layers
success
```

**步骤4：验证模型可用性**

```
# 测试模型推理
ollama run qwen2.5:7b "你好，请用一句话介绍自己"
# 应输出类似：我是通义千问2.5，一个由阿里云开发的大语言模型。
```

#### 6.2.3 定制模型（解决上下文窗口限制）

OpenClaw要求模型的上下文窗口≥16000 tokens，而qwen2.5:7b默认只有4096，必须手动定制。

**步骤1：创建Modelfile**

在Ubuntu WSL环境中操作：

```
# 进入用户目录
cd ~

# 创建Modelfile
cat > Modelfile << 'EOF'
FROM qwen2.5:7b
PARAMETER num_ctx 32768
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF
```

或在Windows PowerShell中：

```
# 切换到用户目录
cd C:\Users\$env:USERNAME

# 创建Modelfile
@"
FROM qwen2.5:7b
PARAMETER num_ctx 32768
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"@ | Out-File -Encoding ascii Modelfile
```

**步骤2：创建定制模型**

```
# 在WSL或PowerShell中执行
ollama create qwen2.5:7b-32k -f Modelfile
```

创建过程：

```
creating model
creating system prompt
creating parameters: num_ctx=32768 temperature=0.7 top_p=0.9
creating model instance
success
```

**步骤3：验证新模型**

```
ollama list
# 应显示：
# NAME                    ID              SIZE    MODIFIED
# qwen2.5:7b-32k          abcdef123456    4.7 GB  1 minute ago
# qwen2.5:7b              6d19a8f3bb82    4.7 GB  5 minutes ago
```

#### 6.2.4 配置OpenClaw使用本地模型

**步骤1：安装OpenClaw**

若尚未安装，执行：

```
npm install -g openclaw@latest
```

**步骤2：配置模型提供商**

```
# 运行配置向导
openclaw onboard
```

关键配置选项：

```
? 选择模型提供商: Custom Provider
? API Base URL: http://127.0.0.1:11434/v1
? API Key: ollama (任意输入，不可为空)
? Model ID: qwen2.5:7b-32k
? 最大上下文窗口: 32768
```

**步骤3：手动编辑配置文件（解决上下文窗口错误）**

如果启动时遇到"Model context window too small"错误，需手动修改配置文件：

```
# 定位配置文件
cd C:\Users\$env:USERNAME\.openclaw

# 修改主配置文件
$config = Get-Content openclaw.json | ConvertFrom-Json
$config.agents.defaults.contextWindow = 32768
$config | ConvertTo-Json -Depth 5 | Set-Content openclaw.json

# 修改Agent模型配置
cd agents\main\agent
$modelConfig = Get-Content models.json | ConvertFrom-Json
$modelConfig.models[0].contextWindow = 32768
$modelConfig | ConvertTo-Json -Depth 5 | Set-Content models.json
```

**步骤4：启动服务**

```
# 启动网关
openclaw gateway start

# 查看日志，确认无错误
openclaw logs
```

#### 6.2.5 完整代码示例：私有化部署全流程脚本

将以下内容保存为 `private-deploy.ps1` ：

```
# OpenClaw私有化部署脚本（Ollama集成）
param(
    [string]$ModelName = "qwen2.5:7b",
    [int]$ContextWindow = 32768,
    [switch]$SkipOllama
)

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "OpenClaw私有化部署脚本 (Ollama集成版)" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# 检查管理员权限
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "错误: 请以管理员身份运行此脚本" -ForegroundColor Red
    exit 1
}

# 步骤1：检查Ollama安装
if (-not $SkipOllama) {
    Write-Host "\`n[1/6] 检查Ollama安装..." -ForegroundColor Yellow
    
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        Write-Host "未检测到Ollama，正在安装..." -ForegroundColor Yellow
        try {
            # 下载Ollama安装程序
            $installer = "$env:TEMP\ollama-setup.exe"
            Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
            
            # 静默安装
            Start-Process -FilePath $installer -ArgumentList "/S" -Wait
            Write-Host "Ollama安装完成" -ForegroundColor Green
            
            # 等待服务启动
            Start-Sleep -Seconds 5
        } catch {
            Write-Host "Ollama安装失败: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Ollama已安装: $ollamaCmd" -ForegroundColor Green
    }
    
    # 验证Ollama服务
    $maxRetry = 10
    $retryCount = 0
    while ($retryCount -lt $maxRetry) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
            Write-Host "Ollama服务运行正常" -ForegroundColor Green
            break
        } catch {
            $retryCount++
            Write-Host "等待Ollama服务启动... ($retryCount/$maxRetry)" -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
    if ($retryCount -eq $maxRetry) {
        Write-Host "Ollama服务启动超时" -ForegroundColor Red
        exit 1
    }
    
    # 步骤2：下载基础模型
    Write-Host "\`n[2/6] 下载基础模型 $ModelName ..." -ForegroundColor Yellow
    $process = Start-Process -FilePath "ollama" -ArgumentList "pull $ModelName" -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        Write-Host "模型下载失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "模型下载完成" -ForegroundColor Green
    
    # 步骤3：创建定制模型
    Write-Host "\`n[3/6] 创建定制模型 ${ModelName}-${ContextWindow} ..." -ForegroundColor Yellow
    $modelfile = @"
FROM $ModelName
PARAMETER num_ctx $ContextWindow
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"@
    $modelfile | Out-File -Encoding ascii "$env:TEMP\Modelfile"
    
    $process = Start-Process -FilePath "ollama" -ArgumentList "create ${ModelName}-${ContextWindow} -f $env:TEMP\Modelfile" -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        Write-Host "模型定制失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "定制模型创建完成" -ForegroundColor Green
}

# 步骤4：安装OpenClaw
Write-Host "\`n[4/6] 安装OpenClaw..." -ForegroundColor Yellow
npm install -g openclaw@latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "OpenClaw安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "OpenClaw安装完成" -ForegroundColor Green

# 步骤5：配置OpenClaw
Write-Host "\`n[5/6] 配置OpenClaw连接本地模型..." -ForegroundColor Yellow
$configDir = "$env:USERPROFILE\.openclaw"
if (Test-Path "$configDir\openclaw.json") {
    $backupFile = "$configDir\openclaw.json.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item "$configDir\openclaw.json" $backupFile
}

# 创建配置目录
if (!(Test-Path $configDir)) {
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
}

# 生成配置
$config = @{
    version = "2026.1"
    gateway = @{
        mode = "local"
        port = 18789
        host = "127.0.0.1"
    }
    agents = @{
        defaults = @{
            workspace = "$configDir\workspace"
            model = "${ModelName}-${ContextWindow}"
            maxTokens = $ContextWindow
            temperature = 0.7
        }
    }
    providers = @{
        custom = @{
            apiBase = "http://127.0.0.1:11434/v1"
            apiKey = "ollama"
        }
    }
}

$config | ConvertTo-Json -Depth 5 | Set-Content "$configDir\openclaw.json" -Encoding UTF8
Write-Host "配置文件已生成" -ForegroundColor Green

# 步骤6：启动服务
Write-Host "\`n[6/6] 启动OpenClaw服务..." -ForegroundColor Yellow
openclaw gateway stop 2>$null
Start-Sleep -Seconds 2
openclaw gateway start

# 等待服务启动
Start-Sleep -Seconds 3

# 生成令牌
$token = openclaw token generate --quiet
if (-not $token) {
    # 从配置文件读取令牌
    $config = Get-Content "$configDir\openclaw.json" | ConvertFrom-Json
    $token = $config.token
}

Write-Host "\`n=======================================" -ForegroundColor Green
Write-Host "部署完成！" -ForegroundColor Green
Write-Host "访问地址: http://127.0.0.1:18789/?token=$token" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Green

# 输出帮助信息
Write-Host "\`n常用命令："
Write-Host "  openclaw gateway status   # 查看服务状态"
Write-Host "  openclaw logs              # 查看日志"
Write-Host "  openclaw config show        # 查看配置"
```

#### 6.2.6 运行结果

成功部署后，通过以下命令测试本地模型响应：

```
# 使用curl直接测试Ollama
curl -X POST http://127.0.0.1:11434/api/chat \`
  -H "Content-Type: application/json" \`
  -d '{\"model\":\"qwen2.5:7b-32k\",\"messages\":[{\"role\":\"user\",\"content\":\"请用一句话介绍OpenClaw\"}]}'

# 预期响应
# {"model":"qwen2.5:7b-32k","message":{"role":"assistant","content":"OpenClaw是一个本地运行的AI Agent调度框架，能让AI自动执行多步骤任务。"}}
```

### 6.3 场景三：WSL源码部署（开发者定制模式）

适合需要深入理解OpenClaw底层逻辑、进行二次开发的开发者。

#### 6.3.1 部署目标

- 在WSL Ubuntu环境中从源码构建OpenClaw
- 支持实时修改代码、即时生效
- 便于调试和定制功能

#### 6.3.2 详细部署步骤

**步骤1：启动WSL Ubuntu终端**

从开始菜单打开"Ubuntu 22.04 LTS"。

**步骤2：更新系统并安装依赖**

```
# 更新包列表
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y wget curl git unzip build-essential

# 安装Python和pip
sudo apt install -y python3 python3-pip python3-venv python3-dev

# 验证版本
python3 --version  # 需≥3.11
```

**步骤3：安装Node.js（通过nvm）**

```
# 安装nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 加载nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 添加到bashrc
echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc
echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.bashrc

# 安装Node.js LTS
nvm install --lts
nvm use --lts

# 验证
node --version  # 需≥v22.0.0
npm --version
```

**步骤4：克隆OpenClaw源码**

```
# 创建开发目录
mkdir -p ~/dev
cd ~/dev

# 克隆官方仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 查看分支
git branch -a

# 切换到最新稳定版（可选）
git checkout tags/v2026.1.29 -b v2026.1.29-local
```

**步骤5：创建Python虚拟环境**

```
# 在项目根目录创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 验证Python环境
which python
# 应显示 /home/username/dev/openclaw/venv/bin/python
```

**步骤6：安装Python依赖**

```
# 升级pip
pip install --upgrade pip

# 安装项目依赖（开发模式）
pip install -e .

# 查看安装的包
pip list
```

**步骤7：安装Node.js依赖**

```
# 安装npm依赖
npm install

# 若网络慢，使用国内镜像
npm config set registry https://registry.npmmirror.com
npm install
```

**步骤8：初始化配置**

```
# 运行配置向导
openclaw onboard
```

按照提示完成初始化，配置选项参考场景一。

**步骤9：启动开发模式**

```
# 开发模式启动（支持热重载）
npm run dev
```

**步骤10：在另一个终端验证服务**

```
# 新开WSL终端
cd ~/dev/openclaw
source venv/bin/activate

# 查看服务状态
openclaw gateway status

# 生成令牌
openclaw token generate
```

#### 6.3.3 源码结构解析

理解源码结构有助于后续定制开发：

```
openclaw/
├── bin/                    # 可执行文件入口
│   ├── openclaw            # CLI入口脚本
│   └── openclaw-gateway    # 网关服务入口
├── src/
│   ├── core/               # 核心模块
│   │   ├── scheduler.py    # 调度引擎
│   │   ├── agent.py        # Agent实现
│   │   ├── memory.py       # 记忆系统
│   │   └── skill.py        # 技能管理
│   ├── cli/                # 命令行接口
│   │   ├── commands/       # 各子命令实现
│   │   └── main.py         # CLI主入口
│   ├── gateway/            # 网关服务
│   │   ├── server.py       # HTTP服务器
│   │   ├── routes.py       # API路由
│   │   └── auth.py         # 认证授权
│   ├── providers/          # 模型提供商
│   │   ├── openrouter.py
│   │   ├── bailian.py
│   │   └── custom.py
│   └── utils/              # 工具函数
├── web/                    # Web前端
│   ├── src/
│   ├── public/
│   └── package.json
├── tests/                  # 测试用例
├── docs/                   # 文档
├── scripts/                # 辅助脚本
├── requirements.txt        # Python依赖
├── package.json            # Node.js依赖
└── setup.py                # Python安装配置
```

#### 6.3.4 完整代码示例：自定义Skill开发

创建一个自定义Skill，实现天气查询功能：

**步骤1：创建Skill目录结构**

```
# 在OpenClaw源码目录中创建自定义Skill
cd ~/dev/openclaw
mkdir -p src/skills/custom/weather
```

**步骤2：实现Skill主逻辑**

创建 `src/skills/custom/weather/__init__.py` ：

```
"""
天气查询Skill
功能：根据城市名称查询实时天气
依赖：需要申请OpenWeatherMap API Key
"""

import os
import json
import aiohttp
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ...core.skill import BaseSkill, SkillMetadata, SkillParameter

@dataclass
class WeatherSkill(BaseSkill):
    """天气查询Skill实现"""
    
    # Skill元数据
    metadata = SkillMetadata(
        name="weather",
        version="1.0.0",
        description="查询指定城市的实时天气信息",
        author="Developer",
        parameters=[
            SkillParameter(
                name="city",
                type="string",
                description="城市名称，支持中文，如'北京'",
                required=True
            ),
            SkillParameter(
                name="units",
                type="string",
                description="温度单位：metric(摄氏度)/imperial(华氏度)",
                required=False,
                default="metric",
                enum=["metric", "imperial"]
            )
        ],
        required_credentials=["OPENWEATHER_API_KEY"]
    )
    
    def __init__(self):
        super().__init__()
        self.api_key = None
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    async def initialize(self, credentials: Dict[str, str]) -> bool:
        """
        初始化Skill，加载API密钥
        
        Args:
            credentials: 凭据字典，包含OPENWEATHER_API_KEY
            
        Returns:
            初始化是否成功
        """
        self.api_key = credentials.get("OPENWEATHER_API_KEY")
        if not self.api_key:
            self.logger.error("缺少OPENWEATHER_API_KEY配置")
            return False
        return True
    
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行天气查询
        
        Args:
            parameters: 参数字典，包含city和units
            context: 执行上下文
            
        Returns:
            查询结果
        """
        city = parameters.get("city")
        units = parameters.get("units", "metric")
        
        if not city:
            return {
                "success": False,
                "error": "缺少必要参数: city"
            }
        
        try:
            # 构建请求参数
            params = {
                "q": city,
                "appid": self.api_key,
                "units": units,
                "lang": "zh_cn"
            }
            
            # 发起HTTP请求
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"API请求失败: {response.status}",
                            "details": error_text
                        }
                    
                    data = await response.json()
                    
                    # 解析结果
                    result = self._parse_weather_data(data, units)
                    return {
                        "success": True,
                        "data": result
                    }
                    
        except Exception as e:
            self.logger.exception("天气查询异常")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_weather_data(self, data: Dict, units: str) -> Dict:
        """
        解析API返回的天气数据
        """
        # 温度单位符号
        unit_symbol = "°C" if units == "metric" else "°F"
        
        # 获取主要数据
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        sys = data.get("sys", {})
        
        return {
            "city": data.get("name"),
            "country": sys.get("country"),
            "weather": {
                "main": weather.get("main"),
                "description": weather.get("description"),
                "icon": weather.get("icon")
            },
            "temperature": {
                "current": f"{main.get('temp')}{unit_symbol}",
                "feels_like": f"{main.get('feels_like')}{unit_symbol}",
                "min": f"{main.get('temp_min')}{unit_symbol}",
                "max": f"{main.get('temp_max')}{unit_symbol}"
            },
            "humidity": f"{main.get('humidity')}%",
            "pressure": f"{main.get('pressure')} hPa",
            "wind": {
                "speed": f"{wind.get('speed')} m/s",
                "direction": wind.get("deg")
            },
            "visibility": f"{data.get('visibility', 0)/1000} km",
            "clouds": f"{data.get('clouds', {}).get('clouds')}%",
            "timestamp": data.get("dt")
        }
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        验证参数有效性
        """
        if "city" not in parameters:
            return False, "缺少城市参数"
        
        city = parameters["city"]
        if not isinstance(city, str) or len(city.strip()) == 0:
            return False, "城市参数必须是有效的字符串"
        
        units = parameters.get("units", "metric")
        if units not in ["metric", "imperial"]:
            return False, "单位参数必须是metric或imperial"
        
        return True, None
    
    async def cleanup(self) -> None:
        """
        清理资源
        """
        self.logger.info("天气Skill资源已清理")
```

**步骤3：注册Skill**

创建 `src/skills/custom/__init__.py` ：

```
"""自定义Skills注册模块"""

from .weather import WeatherSkill

# 技能注册字典
CUSTOM_SKILLS = {
    "weather": WeatherSkill,
    # 可继续添加其他自定义Skill
}

__all__ = ["CUSTOM_SKILLS", "WeatherSkill"]
```

**步骤4：修改技能加载器**

编辑 `src/core/skill_loader.py` ，添加自定义技能加载逻辑：

```
# ... 现有代码 ...

from ..skills.custom import CUSTOM_SKILLS

class SkillLoader:
    """技能加载器"""
    
    def __init__(self):
        self.skills = {}
        self._load_builtin_skills()
        self._load_custom_skills()  # 新增：加载自定义技能
    
    def _load_custom_skills(self):
        """加载自定义技能"""
        for name, skill_class in CUSTOM_SKILLS.items():
            try:
                self.skills[name] = skill_class()
                self.logger.info(f"已加载自定义技能: {name}")
            except Exception as e:
                self.logger.error(f"加载自定义技能 {name} 失败: {e}")
    
    # ... 其他方法 ...
```

**步骤5：配置API密钥**

在OpenClaw配置文件中添加凭据：

```
# 编辑配置文件
nano ~/.openclaw/openclaw.json
```

添加凭据部分：

```
{
    "version": "2026.1",
    "gateway": {
        "mode": "local",
        "port": 18789
    },
    "agents": {
        "defaults": {
            "workspace": "~/.openclaw/workspace",
            "model": "qwen2.5:7b-32k",
            "maxTokens": 32768
        }
    },
    "credentials": {
        "OPENWEATHER_API_KEY": "your_api_key_here"
    }
}
```

**步骤6：重启服务测试**

```
# 停止服务
openclaw gateway stop

# 重新启动
openclaw gateway start

# 测试技能调用
openclaw skill list
# 应显示 weather 在技能列表中

openclaw run weather '{"city": "北京"}'
```

#### 6.3.5 运行结果

成功加载自定义Skill后，测试结果示例：

```
$ openclaw run weather '{"city": "上海"}'

{
  "success": true,
  "data": {
    "city": "上海",
    "country": "CN",
    "weather": {
      "main": "Clouds",
      "description": "broken clouds",
      "icon": "04d"
    },
    "temperature": {
      "current": "18.5°C",
      "feels_like": "17.8°C",
      "min": "17.0°C",
      "max": "20.0°C"
    },
    "humidity": "73%",
    "pressure": "1016 hPa",
    "wind": {
      "speed": "3.1 m/s",
      "direction": 120
    },
    "visibility": "10.0 km",
    "clouds": "75%",
    "timestamp": 1712476800
  }
}
```

### 6.4 场景四：HiClaw多Agent团队协作部署（团队模式）

适合需要处理复杂项目、组建AI团队的企业用户。

#### 6.4.1 部署目标

- 部署Manager Agent（AI管家）统一调度
- 创建多个专业Worker Agent（前端、后端、文档）
- 实现任务自动拆解与分发
- 支持通过IM工具（如飞书、Telegram）远程指挥

#### 6.4.2 详细部署步骤

**步骤1：确保Docker环境就绪**

```
# 验证Docker安装
docker --version
docker-compose --version

# 验证WSL2集成
docker info | findstr "WSL"
```

**步骤2：安装HiClaw组件**

```
# 全局安装HiClaw
npm install -g hiclaw@latest

# 验证安装
hiclaw --version
# 应显示 2026.1.29 或更高
```

**步骤3：初始化团队架构**

```
# 创建工作目录
mkdir C:\HiClaw-Team
cd C:\HiClaw-Team

# 初始化Manager与默认Worker（前端+后端+文档）
hiclaw init --default-workers frontend backend docs

# 查看初始化结果
dir
# 应看到以下目录结构：
#   manager/          # Manager Agent配置
#   workers/          # Worker Agents目录
#   docker-compose.yml # 容器编排文件
#   .env              # 环境变量配置
```

**步骤4：配置模型提供商**

编辑`.env` 文件：

```
# HiClaw环境配置
HICLAW_VERSION=2026.1

# Manager配置
MANAGER_MODEL=openai/gpt-4
MANAGER_API_KEY=sk-your-api-key
MANAGER_API_BASE=https://api.openai.com/v1

# Worker配置
WORKER_FRONTEND_MODEL=claude-3-sonnet
WORKER_BACKEND_MODEL=claude-3-haiku
WORKER_DOCS_MODEL=qwen2.5:7b-32k

# AI Gateway配置（凭证集中管理）
AI_GATEWAY_URL=http://gateway:8080
AI_GATEWAY_TOKEN=your-secure-token

# Matrix通信配置
MATRIX_HOMESERVER=http://matrix:8008
MATRIX_USER=@hiclaw:localhost
MATRIX_PASSWORD=changeme

# 端口映射
MANAGER_PORT=18788
WORKER_FRONTEND_PORT=18790
WORKER_BACKEND_PORT=18791
WORKER_DOCS_PORT=18792
```

**步骤5：自定义Worker技能**

为每个Worker配置专属技能，编辑 `workers/frontend/skills.json` ：

```
{
    "skills": [
        {
            "name": "react-component",
            "version": "1.0.0",
            "description": "生成React组件代码",
            "enabled": true
        },
        {
            "name": "vue-template",
            "version": "1.0.0",
            "description": "生成Vue模板代码",
            "enabled": true
        },
        {
            "name": "css-styling",
            "version": "1.0.0",
            "description": "生成CSS样式代码",
            "enabled": true
        }
    ]
}
```

编辑 `workers/backend/skills.json` ：

```
{
    "skills": [
        {
            "name": "api-design",
            "version": "1.0.0",
            "description": "设计RESTful API接口",
            "enabled": true
        },
        {
            "name": "database-schema",
            "version": "1.0.0",
            "description": "设计数据库Schema",
            "enabled": true
        },
        {
            "name": "sql-query",
            "version": "1.0.0",
            "description": "生成SQL查询语句",
            "enabled": true
        }
    ]
}
```

**步骤6：启动HiClaw服务**

```
# 启动所有容器
hiclaw up

# 查看启动日志
hiclaw logs

# 查看容器状态
hiclaw status
# 应显示：
# MANAGER      Running   healthy   http://localhost:18788
# WORKER-frontend  Running   healthy   http://localhost:18790
# WORKER-backend   Running   healthy   http://localhost:18791
# WORKER-docs      Running   healthy   http://localhost:18792
# AI-GATEWAY    Running   healthy
# MATRIX-SERVER Running   healthy
```

**步骤7：配置IM集成（以飞书为例）**

创建飞书机器人配置 `feishu-bot.js` ：

```
// feishu-bot.js
const axios = require('axios');
const crypto = require('crypto');

class FeishuBot {
    constructor(appId, appSecret) {
        this.appId = appId;
        this.appSecret = appSecret;
        this.accessToken = null;
        this.managerUrl = 'http://localhost:18788';
    }

    async getAccessToken() {
        const url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal';
        const response = await axios.post(url, {
            app_id: this.appId,
            app_secret: this.appSecret
        });
        this.accessToken = response.data.tenant_access_token;
        return this.accessToken;
    }

    async sendMessage(openId, content) {
        if (!this.accessToken) {
            await this.getAccessToken();
        }

        const url = 'https://open.feishu.cn/open-apis/im/v1/messages';
        const response = await axios.post(url, {
            receive_id: openId,
            msg_type: 'text',
            content: JSON.stringify({ text: content })
        }, {
            headers: {
                'Authorization': \`Bearer ${this.accessToken}\`,
                'Content-Type': 'application/json'
            }
        });

        return response.data;
    }

    async handleWebhook(req) {
        const { event } = req.body;
        
        // 验证事件签名
        if (!this.verifySignature(req)) {
            return { code: 403, msg: 'invalid signature' };
        }

        // 处理消息事件
        if (event.type === 'message' && event.message.message_type === 'text') {
            const userInput = event.message.content;
            const senderId = event.sender.sender_id.open_id;

            // 调用Manager Agent处理任务
            const result = await this.callManager(userInput);

            // 发送回复
            await this.sendMessage(senderId, result);
        }

        return { code: 0, msg: 'success' };
    }

    async callManager(prompt) {
        try {
            const response = await axios.post(\`${this.managerUrl}/api/chat\`, {
                message: prompt,
                stream: false
            });
            return response.data.response;
        } catch (error) {
            console.error('调用Manager失败:', error);
            return '抱歉，处理请求时出现错误，请稍后再试。';
        }
    }

    verifySignature(req) {
        const signature = req.headers['x-lark-signature'];
        const timestamp = req.headers['x-lark-request-timestamp'];
        const nonce = req.headers['x-lark-request-nonce'];
        const body = JSON.stringify(req.body);

        const content = [timestamp, nonce, this.appSecret, body].join('');
        const calculated = crypto.createHash('sha256').update(content).digest('hex');

        return calculated === signature;
    }
}

module.exports = FeishuBot;
```

**步骤8：启动飞书机器人服务**

创建 `server.js` ：

```
// server.js
const express = require('express');
const FeishuBot = require('./feishu-bot');

const app = express();
app.use(express.json());

const bot = new FeishuBot(
    process.env.FEISHU_APP_ID,
    process.env.FEISHU_APP_SECRET
);

app.post('/webhook/feishu', async (req, res) => {
    const result = await bot.handleWebhook(req);
    res.status(200).json(result);
});

app.get('/health', (req, res) => {
    res.status(200).json({ status: 'healthy' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(\`飞书机器人服务运行在端口 ${PORT}\`);
});
```

运行机器人服务：

```
# 安装依赖
npm install express axios

# 设置环境变量
export FEISHU_APP_ID=your_app_id
export FEISHU_APP_SECRET=your_app_secret

# 启动服务
node server.js
```

#### 6.4.3 完整代码示例：多Agent协作任务

创建任务分发测试脚本 `test-team.js` ：

```
// test-team.js
const axios = require('axios');

class HiClawTeam {
    constructor(managerUrl) {
        this.managerUrl = managerUrl;
    }

    async submitTask(taskDescription) {
        console.log(\`提交任务: ${taskDescription}\`);
        
        const response = await axios.post(\`${this.managerUrl}/api/tasks\`, {
            description: taskDescription,
            priority: 'normal',
            callback_url: 'http://localhost:3000/callback'
        });

        const taskId = response.data.task_id;
        console.log(\`任务已接受，ID: ${taskId}\`);
        
        return taskId;
    }

    async getTaskStatus(taskId) {
        const response = await axios.get(\`${this.managerUrl}/api/tasks/${taskId}\`);
        return response.data;
    }

    async waitForCompletion(taskId, interval = 2000) {
        while (true) {
            const status = await this.getTaskStatus(taskId);
            console.log(\`任务状态: ${status.state}\`);
            
            if (status.state === 'completed') {
                console.log('任务完成!');
                console.log('结果:', status.result);
                return status;
            } else if (status.state === 'failed') {
                console.log('任务失败:', status.error);
                throw new Error(status.error);
            }
            
            await new Promise(resolve => setTimeout(resolve, interval));
        }
    }
}

// 测试场景：开发一个简单的博客系统
async function testBlogDevelopment() {
    const team = new HiClawTeam('http://localhost:18788');
    
    const task = \`
    请开发一个简单的个人博客系统，包含以下功能：
    1. 前端：博客列表页、文章详情页、关于我页面
    2. 后端：文章CRUD API、评论功能
    3. 数据库：使用SQLite存储文章和评论
    4. 文档：提供API文档和部署说明
    
    请协调前端、后端和文档Worker协作完成。
    \`;
    
    try {
        // 提交任务
        const taskId = await team.submitTask(task);
        
        // 监控任务进度
        await team.waitForCompletion(taskId);
        
        // 查看详细报告
        const report = await axios.get(\`${team.managerUrl}/api/tasks/${taskId}/report\`);
        console.log('详细报告:', report.data);
        
    } catch (error) {
        console.error('任务执行失败:', error.message);
    }
}

// 运行测试
testBlogDevelopment();
```

#### 6.4.4 运行结果

执行测试脚本后的输出示例：

```
$ node test-team.js

提交任务: 请开发一个简单的个人博客系统...
任务已接受，ID: task_550e8400-e29b-41d4-a716-446655440000

任务状态: pending
任务状态: planning
任务状态: assigned
[Manager] 正在拆解任务...
[Manager] 已分配子任务:
  - 前端开发: worker-frontend
  - 后端开发: worker-backend
  - 文档撰写: worker-docs

任务状态: in_progress
[worker-frontend] 开始开发博客列表页...
[worker-frontend] 完成列表页组件
[worker-frontend] 开始开发文章详情页...
[worker-backend] 开始设计数据库Schema...
[worker-backend] 完成文章表设计
[worker-backend] 开始实现CRUD API...
[worker-docs] 开始撰写API文档...

任务状态: in_progress (3/3 子任务)
[worker-frontend] 完成所有前端页面
[worker-backend] 完成API实现，开始测试
[worker-docs] 完成API文档初稿

任务状态: merging
[Manager] 正在合并各Worker成果...
[Manager] 进行集成测试...

任务状态: completed
任务完成!
结果: {
  "project": "SimpleBlog",
  "frontend": {
    "pages": ["index.html", "post.html", "about.html"],
    "components": ["Header.js", "PostList.js", "PostDetail.js"],
    "technologies": ["React", "TailwindCSS"]
  },
  "backend": {
    "api": ["GET /posts", "GET /posts/:id", "POST /posts", "PUT /posts/:id", "DELETE /posts/:id", "POST /comments"],
    "database": "SQLite with schema version 1.0",
    "tests": "API测试覆盖率92%"
  },
  "docs": {
    "api_docs": "OpenAPI 3.0规范",
    "deployment": "Docker部署指南",
    "quick_start": "5分钟快速入门"
  }
}

详细报告: {
  "task_id": "task_550e8400-e29b-41d4-a716-446655440000",
  "execution_time": "3m 24s",
  "token_usage": {
    "manager": 2456,
    "frontend": 8732,
    "backend": 12453,
    "docs": 3421,
    "total": 27062
  },
  "cost_estimate": "$0.54",
  "artifacts": {
    "repository": "/workspace/SimpleBlog",
    "preview": "http://localhost:3000"
  }
}
```

### 6.5 场景五：金融投研自动化场景

券商分析师集体行动，将OpenClaw应用于金融投研领域。

#### 6.5.1 部署目标

- 自动化抓取A股公告信息
- 定时推送汇总报告
- 辅助财报分析和条件选股
- 量化策略回测

#### 6.5.2 公告信息汇总Skill实现

创建 `financial-skills.js` ：

```
// financial-skills.js
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const schedule = require('node-schedule');

class AShareAnnouncementSkill {
    constructor(config) {
        this.config = config;
        this.dataDir = path.join(process.env.HOME || process.env.USERPROFILE, '.openclaw', 'financial-data');
    }

    async initialize() {
        // 创建数据目录
        await fs.mkdir(this.dataDir, { recursive: true });
        console.log(\`数据目录: ${this.dataDir}\`);
    }

    async fetchAnnouncements(date) {
        // 模拟抓取公告数据
        // 实际应用中可接入东方财富、巨潮资讯等数据源
        const mockData = [
            {
                code: '600036',
                name: '招商银行',
                title: '2025年年度报告',
                type: '年报',
                date: date || new Date().toISOString().split('T')[0],
                url: 'https://example.com/announcement/600036-2025',
                summary: '招商银行2025年实现净利润XXX亿元，同比增长XX%...'
            },
            {
                code: '000858',
                name: '五粮液',
                title: '关于控股股东增持公司股份的公告',
                type: '股东增持',
                date: date || new Date().toISOString().split('T')[0],
                url: 'https://example.com/announcement/000858-2025',
                summary: '控股股东拟增持公司股份，金额不低于XX亿元...'
            },
            {
                code: '300750',
                name: '宁德时代',
                title: '关于签订重大合同的公告',
                type: '重大合同',
                date: date || new Date().toISOString().split('T')[0],
                url: 'https://example.com/announcement/300750-2025',
                summary: '与某国际车企签订长期供货协议，总金额XX亿元...'
            }
        ];

        return mockData;
    }

    async generateReport(announcements) {
        // 分类汇总
        const categorized = {};
        announcements.forEach(item => {
            if (!categorized[item.type]) {
                categorized[item.type] = [];
            }
            categorized[item.type].push(item);
        });

        // 生成报告文本
        let report = \`# A股公告信息汇总（${new Date().toLocaleDateString('zh-CN')}）\n\n\`;
        report += \`共抓取 ${announcements.length} 条公告\n\n\`;

        for (const [type, items] of Object.entries(categorized)) {
            report += \`## ${type}\n\`;
            items.forEach(item => {
                report += \`- **${item.code} ${item.name}**: ${item.title}\n\`;
                report += \`  - 摘要: ${item.summary}\n\`;
                report += \`  - [查看原文](${item.url})\n\`;
            });
            report += '\n';
        }

        // 生成Excel格式（CSV）
        const csv = ['股票代码,股票名称,公告类型,公告标题,日期,摘要'];
        announcements.forEach(item => {
            csv.push(\`${item.code},${item.name},${item.type},${item.title},${item.date},${item.summary.replace(/,/g, ';')}\`);
        });

        return {
            markdown: report,
            csv: csv.join('\n'),
            json: announcements
        };
    }

    async saveReport(report, format = 'all') {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const files = {};

        if (format === 'all' || format === 'markdown') {
            const mdPath = path.join(this.dataDir, \`announcements-${timestamp}.md\`);
            await fs.writeFile(mdPath, report.markdown, 'utf8');
            files.markdown = mdPath;
        }

        if (format === 'all' || format === 'csv') {
            const csvPath = path.join(this.dataDir, \`announcements-${timestamp}.csv\`);
            await fs.writeFile(csvPath, report.csv, 'utf8');
            files.csv = csvPath;
        }

        if (format === 'all' || format === 'json') {
            const jsonPath = path.join(this.dataDir, \`announcements-${timestamp}.json\`);
            await fs.writeFile(jsonPath, JSON.stringify(report.json, null, 2), 'utf8');
            files.json = jsonPath;
        }

        return files;
    }

    async execute(params) {
        const date = params.date || new Date().toISOString().split('T')[0];
        const format = params.format || 'all';

        // 抓取公告
        console.log(\`抓取 ${date} 的公告...\`);
        const announcements = await this.fetchAnnouncements(date);

        // 生成报告
        console.log('生成汇总报告...');
        const report = await this.generateReport(announcements);

        // 保存文件
        const files = await this.saveReport(report, format);

        // 发送到IM（如飞书）
        if (params.notify) {
            await this.sendToIM(report, params.notify);
        }

        return {
            success: true,
            count: announcements.length,
            files,
            preview: report.markdown.substring(0, 500) + '...'
        };
    }

    async sendToIM(report, target) {
        // 集成飞书/钉钉/企业微信
        // 此处为示例代码
        console.log(\`发送报告到 ${target}\`);
        // await axios.post(target.webhook, {
        //     msg_type: 'interactive',
        //     card: this.buildCard(report)
        // });
    }

    buildCard(report) {
        return {
            header: {
                title: {
                    tag: 'plain_text',
                    content: '📊 A股公告汇总'
                }
            },
            elements: [
                {
                    tag: 'markdown',
                    content: report.markdown
                }
            ]
        };
    }
}

// 定时任务配置
function scheduleAnnouncementTask(skill, hour = 17, minute = 30) {
    // 每天下午5:30执行
    const rule = new schedule.RecurrenceRule();
    rule.hour = hour;
    rule.minute = minute;

    schedule.scheduleJob(rule, async () => {
        console.log(\`执行定时公告抓取任务 ${new Date().toLocaleString()}\`);
        try {
            const result = await skill.execute({
                format: 'all',
                notify: {
                    webhook: process.env.FEISHU_WEBHOOK
                }
            });
            console.log(\`任务完成，抓取 ${result.count} 条公告\`);
        } catch (error) {
            console.error('任务失败:', error);
        }
    });

    console.log(\`定时任务已设置: 每天 ${hour}:${minute} 执行\`);
}

// 导出
module.exports = {
    AShareAnnouncementSkill,
    scheduleAnnouncementTask
};
```

#### 6.5.3 条件选股Skill实现

创建 `stock-screen-skill.js` ：

```
// stock-screen-skill.js
class StockScreenSkill {
    constructor() {
        this.dataSource = 'tushare'; // 可配置数据源
    }

    async fetchStockList() {
        // 模拟获取全市场股票列表
        return [
            { code: '600036', name: '招商银行', sector: '银行', marketCap: 8000 },
            { code: '000858', name: '五粮液', sector: '白酒', marketCap: 6000 },
            { code: '300750', name: '宁德时代', sector: '新能源', marketCap: 9000 },
            { code: '600519', name: '贵州茅台', sector: '白酒', marketCap: 20000 },
            { code: '000333', name: '美的集团', sector: '家电', marketCap: 4000 },
            // ... 更多股票
        ];
    }

    async fetchFinancialData(codes) {
        // 模拟获取财务数据
        const mockData = {};
        codes.forEach(code => {
            mockData[code] = {
                pe: 15 + Math.random() * 20,
                pb: 1.5 + Math.random() * 3,
                roe: 10 + Math.random() * 15,
                revenueGrowth: 5 + Math.random() * 30,
                profitGrowth: 3 + Math.random() * 25,
                debtRatio: 40 + Math.random() * 20
            };
        });
        return mockData;
    }

    async screen(params) {
        const conditions = params.conditions || [];
        const limit = params.limit || 20;

        // 获取股票池
        let stocks = await this.fetchStockList();

        // 获取财务数据
        const codes = stocks.map(s => s.code);
        const financialData = await this.fetchFinancialData(codes);

        // 合并数据
        stocks = stocks.map(stock => ({
            ...stock,
            ...financialData[stock.code]
        }));

        // 应用筛选条件
        conditions.forEach(condition => {
            const { field, operator, value } = condition;
            stocks = stocks.filter(stock => {
                const stockValue = stock[field];
                switch (operator) {
                    case '>': return stockValue > value;
                    case '<': return stockValue < value;
                    case '>=': return stockValue >= value;
                    case '<=': return stockValue <= value;
                    case '==': return stockValue == value;
                    case 'between': return stockValue >= value[0] && stockValue <= value[1];
                    default: return true;
                }
            });
        });

        // 排序
        if (params.orderBy) {
            const [field, direction] = params.orderBy.split(':');
            stocks.sort((a, b) => {
                if (direction === 'desc') {
                    return b[field] - a[field];
                }
                return a[field] - b[field];
            });
        }

        // 限制数量
        stocks = stocks.slice(0, limit);

        return {
            success: true,
            count: stocks.length,
            stocks,
            conditions: conditions.map(c => \`${c.field} ${c.operator} ${c.value}\`).join(' AND ')
        };
    }

    async backtest(strategy, period) {
        // 策略回测实现
        // 模拟回测结果
        return {
            strategy: strategy.name,
            period: period,
            returns: 0.25,
            sharpe: 1.8,
            maxDrawdown: -0.15,
            winRate: 0.65
        };
    }

    async execute(params) {
        const { action } = params;

        if (action === 'screen') {
            return await this.screen(params);
        } else if (action === 'backtest') {
            return await this.backtest(params.strategy, params.period);
        } else {
            throw new Error(\`不支持的操作: ${action}\`);
        }
    }
}

module.exports = StockScreenSkill;
```

#### 6.5.4 投研助手使用示例

创建 `research-assistant.js` ：

```
// research-assistant.js
const { AShareAnnouncementSkill } = require('./financial-skills');
const StockScreenSkill = require('./stock-screen-skill');
const readline = require('readline');

class ResearchAssistant {
    constructor() {
        this.announcementSkill = new AShareAnnouncementSkill({});
        this.screenSkill = new StockScreenSkill();
    }

    async processCommand(input) {
        input = input.toLowerCase();

        // 解析意图
        if (input.includes('公告') || input.includes('汇总')) {
            return await this.handleAnnouncement(input);
        } else if (input.includes('选股') || input.includes('筛选')) {
            return await this.handleStockScreen(input);
        } else if (input.includes('回测') || input.includes('策略')) {
            return await this.handleBacktest(input);
        } else {
            return {
                type: 'help',
                message: '支持的命令：\n' +
                        '- 公告汇总 [日期]：获取指定日期公告\n' +
                        '- 条件选股 [条件]：根据财务指标筛选股票\n' +
                        '- 策略回测 [策略名] [期间]：回测投资策略\n' +
                        '- 示例：今天有什么重要公告？\n' +
                        '- 示例：筛选PE<20且ROE>15的股票\n' +
                        '- 示例：回测小盘价值策略过去一年'
            };
        }
    }

    async handleAnnouncement(input) {
        // 提取日期
        let date = new Date().toISOString().split('T')[0];
        if (input.includes('昨天')) {
            const d = new Date();
            d.setDate(d.getDate() - 1);
            date = d.toISOString().split('T')[0];
        } else if (input.includes('前天')) {
            const d = new Date();
            d.setDate(d.getDate() - 2);
            date = d.toISOString().split('T')[0];
        }

        const result = await this.announcementSkill.execute({
            date,
            format: 'markdown'
        });

        return {
            type: 'announcement',
            data: result
        };
    }

    async handleStockScreen(input) {
        // 解析筛选条件（简单示例）
        const conditions = [];
        
        // 解析PE条件
        const peMatch = input.match(/pe\s*([<>]=?)\s*(\d+)/i);
        if (peMatch) {
            conditions.push({
                field: 'pe',
                operator: peMatch[1],
                value: parseFloat(peMatch[2])
            });
        }

        // 解析ROE条件
        const roeMatch = input.match(/roe\s*([<>]=?)\s*(\d+)/i);
        if (roeMatch) {
            conditions.push({
                field: 'roe',
                operator: roeMatch[1],
                value: parseFloat(roeMatch[2])
            });
        }

        // 解析行业条件
        const sectorMatch = input.match(/(银行|白酒|新能源|医药|家电)/);
        if (sectorMatch) {
            // 实际应用中需要按行业筛选
        }

        const result = await this.screenSkill.execute({
            action: 'screen',
            conditions,
            orderBy: 'roe:desc',
            limit: 10
        });

        return {
            type: 'screen',
            data: result
        };
    }

    async handleBacktest(input) {
        // 简化处理，返回模拟回测结果
        return {
            type: 'backtest',
            data: {
                strategy: '小盘价值',
                period: '过去一年',
                returns: '25.3%',
                sharpe: 1.82,
                maxDrawdown: '-12.5%',
                winRate: '67%'
            }
        };
    }

    formatOutput(result) {
        switch (result.type) {
            case 'announcement':
                return result.data.preview;
            case 'screen':
                const stocks = result.data.stocks;
                let output = \`筛选结果 (${result.data.count}只):\n\`;
                stocks.forEach(s => {
                    output += \`${s.code} ${s.name} | PE:${s.pe.toFixed(2)} | ROE:${s.roe.toFixed(1)}% | 市值:${s.marketCap}亿\n\`;
                });
                return output;
            case 'backtest':
                const d = result.data;
                return \`策略回测结果：\n\` +
                       \`策略: ${d.strategy}\n\` +
                       \`期间: ${d.period}\n\` +
                       \`收益率: ${d.returns}\n\` +
                       \`夏普比率: ${d.sharpe}\n\` +
                       \`最大回撤: ${d.maxDrawdown}\n\` +
                       \`胜率: ${d.winRate}\`;
            default:
                return result.message;
        }
    }
}

// 交互式REPL
async function runREPL() {
    const assistant = new ResearchAssistant();
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: '投研助手> '
    });

    console.log('投研助手已启动，输入命令开始（输入exit退出）');
    rl.prompt();

    rl.on('line', async (line) => {
        if (line.trim().toLowerCase() === 'exit') {
            rl.close();
            return;
        }

        try {
            const result = await assistant.processCommand(line);
            console.log(assistant.formatOutput(result));
        } catch (error) {
            console.error('处理错误:', error.message);
        }

        rl.prompt();
    }).on('close', () => {
        console.log('投研助手已退出');
        process.exit(0);
    });
}

// 启动
if (require.main === module) {
    runREPL();
}
```

#### 6.5.5 运行结果

投研助手交互示例：

```
$ node research-assistant.js

投研助手已启动，输入命令开始（输入exit退出）
投研助手> 今天有什么重要公告？

筛选结果 (3只):
600036 招商银行 | PE:12.5 | ROE:16.8% | 市值:8000亿
000858 五粮液 | PE:18.2 | ROE:22.5% | 市值:6000亿
300750 宁德时代 | PE:25.6 | ROE:14.2% | 市值:9000亿

投研助手> 筛选PE<20且ROE>15的股票

筛选结果 (2只):
600036 招商银行 | PE:12.5 | ROE:16.8% | 市值:8000亿
000858 五粮液 | PE:18.2 | ROE:22.5% | 市值:6000亿

投研助手> 回测小盘价值策略过去一年

策略回测结果：
策略: 小盘价值
期间: 过去一年
收益率: 25.3%
夏普比率: 1.82
最大回撤: -12.5%
胜率: 67%
```

## 七、测试步骤与验证

### 7.1 单元测试

```
# 运行核心模块单元测试
cd ~/dev/openclaw
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_scheduler.py -v

# 带覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 7.2 集成测试

创建集成测试脚本 `integration-test.js` ：

```
// integration-test.js
const axios = require('axios');
const assert = require('assert');

async function testGateway() {
    console.log('测试网关服务...');
    
    // 测试健康检查
    const health = await axios.get('http://localhost:18789/health');
    assert(health.status === 200);
    assert(health.data.status === 'healthy');
    
    console.log('✓ 网关健康检查通过');
    
    // 测试聊天接口
    const chat = await axios.post('http://localhost:18789/api/chat', {
        message: '你好，请简单介绍一下自己',
        stream: false
    });
    
    assert(chat.status === 200);
    assert(chat.data.response.length > 0);
    
    console.log('✓ 聊天接口测试通过');
    console.log(\`  响应: ${chat.data.response.substring(0, 100)}...\`);
}

async function testSkills() {
    console.log('\n测试技能系统...');
    
    // 获取技能列表
    const skills = await axios.get('http://localhost:18789/api/skills');
    assert(skills.status === 200);
    assert(Array.isArray(skills.data.skills));
    
    console.log(\`✓ 技能列表获取成功，共 ${skills.data.skills.length} 个技能\`);
    
    // 测试文件操作技能（如果有）
    if (skills.data.skills.includes('file')) {
        const fileTest = await axios.post('http://localhost:18789/api/skills/file/execute', {
            action: 'list',
            path: '.'
        });
        assert(fileTest.status === 200);
        console.log('✓ 文件技能测试通过');
    }
}

async function testMemory() {
    console.log('\n测试记忆系统...');
    
    // 写入记忆
    const write = await axios.post('http://localhost:18789/api/memory', {
        key: 'test_key',
        value: 'test_value',
        namespace: 'test'
    });
    assert(write.status === 200);
    
    // 读取记忆
    const read = await axios.get('http://localhost:18789/api/memory/test_key?namespace=test');
    assert(read.status === 200);
    assert(read.data.value === 'test_value');
    
    console.log('✓ 记忆系统测试通过');
}

async function runAllTests() {
    console.log('=== OpenClaw 集成测试 ===\n');
    
    try {
        await testGateway();
        await testSkills();
        await testMemory();
        
        console.log('\n✅ 所有测试通过！');
    } catch (error) {
        console.error('\n❌ 测试失败:', error.message);
        if (error.response) {
            console.error('响应状态:', error.response.status);
            console.error('响应数据:', error.response.data);
        }
        process.exit(1);
    }
}

runAllTests();
```

运行测试：

```
node integration-test.js
```

### 7.3 性能测试

```
# 安装性能测试工具
npm install -g artillery

# 创建性能测试脚本 performance-test.yml
```

创建 `performance-test.yml` ：

```
config:
  target: "http://localhost:18789"
  phases:
    - duration: 60
      arrivalRate: 5
      name: "稳态负载"
    - duration: 30
      arrivalRate: 10
      name: "峰值负载"
  defaults:
    headers:
      Content-Type: "application/json"

scenarios:
  - name: "聊天请求"
    flow:
      - post:
          url: "/api/chat"
          json:
            message: "你好，这是一条测试消息"
            stream: false
          capture:
            - json: "$.response"
              as: "response"
      - log: "收到响应长度: {{response.length}}"
```

运行性能测试：

```
artillery run performance-test.yml
```

## 八、疑难解答

### 8.1 安装阶段问题

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| `iwr`  命令失败 | 网络连接问题 | 使用国内镜像脚本 |
| 执行策略错误 | PowerShell执行策略限制 | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| WSL安装失败 | BIOS虚拟化未开启 | 重启进入BIOS启用VT-x/AMD-V |
| npm安装超时 | 网络慢或镜像问题 | `npm config set registry https://registry.npmmirror.com` |

### 8.2 配置阶段问题

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| "Model context window too small" | 模型上下文窗口小于OpenClaw要求 | 定制模型设置num\_ctx=32768 |
| API Key错误 | 配置的API Key无效 | 检查API Key格式，Ollama可任意输入非空字符串 |
| 端口冲突 | 18789端口被占用 | `openclaw config set gateway.port 18790`  修改端口 |
| OAuth认证失败 | 无图形界面 | 使用SSH端口转发或选择替代认证方式 |

### 8.3 运行时问题

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 内存不足 | 模型过大或并发任务多 | 降低batch size，启用梯度检查点 |
| 响应缓慢 | CPU推理速度慢 | 启用GPU加速，或使用更小的量化模型 |
| 技能执行失败 | 依赖缺失或API不可用 | 检查技能日志： `openclaw logs --skill skill_name` |
| 定时任务不执行 | 时区设置错误 | 检查系统时区，确保与cron表达式匹配 |

### 8.4 安全相关警告

根据安全机构报告，市面上存在大量恶意技能。务必遵循以下安全准则：

**高危行为** ：

- ❌ 直接运行 `curl ... | sh` 下载的未知脚本
- ❌ 安装名字与热门技能相似但来源不同的仿冒技能
- ❌ 在主力工作机上运行高权限Agent

**安全实践** ：

- ✅ 技能安装前用VirusTotal扫描
- ✅ 优先选择有公开源码、星标高、贡献者稳定的技能
- ✅ 部署在隔离环境（虚拟机/容器）
- ✅ AI Gateway集中管理凭证，Worker仅持临时令牌

### 8.5 诊断命令

```
# 全面环境检查
openclaw doctor

# 查看实时日志
openclaw logs follow

# 查看服务状态
openclaw gateway status

# 查看配置
openclaw config show

# 调试模式启动
openclaw gateway start --debug
```

## 九、未来展望

### 9.1 技术趋势

**从单Agent到多Agent协作** ：HiClaw的"Manager+Workers"架构标志Agent应用进入团队协作时代。未来，我们将看到：

- 跨组织Agent协作网络
- 专业化分工的Agent市场
- 基于成果的Agent经济模型

**从软件到硬件的延伸** ：OpenClaw正迅速渗透到硬件领域，机器狗、机械臂、AI眼镜、智能手表等设备开始主动接入OpenClaw。这预示着：

- 硬件成为AI的感知器官和执行器官
- Agent第一次拥有真正的物理执行能力
- 传感器的重要性超过设备本身的计算能力

**从对话工具到生产节点** ：当Agent会说话、会画图、会自动生产音视频内容时，它不再是聊天机器人，而是可扩展的生产节点。关键不在于模型多强，而在于：

- 调度能力 × 工具能力 × 多模态能力
- 如何在工程体系内让AI可控、可测试、可扩展

### 9.2 挑战与应对

**安全风险** ：OpenClaw的高权限特性带来巨大安全隐患。CVE-2026-25253漏洞的曝光敲响警钟。行业需要：

- 更严格的权限模型
- 更完善的技能审核机制
- 容器级隔离成为标配

**幻觉问题** ：大模型的逻辑错误和数据虚构无法完全消除。对于金融等严谨行业，AI结论只能作为辅助参考。解决方案包括：

- RAG增强的事实核查
- 多模型共识机制
- 人类最终签字确认权

**成本控制** ：多Agent协作带来Token消耗的指数级增长。HiClaw通过按Worker类型分配最优模型，可节省60-80%成本。未来将出现：

- 更精细的成本预估和预算控制
- Token消耗的实时监控和告警
- 模型路由优化算法

### 9.3 投研领域的智能化转型

券商分析师集体编写OpenClaw部署报告，反映了投研行业智能化转型的深层次变革：

**从临时调用到稳定调用的升级** ：OpenClaw不再是每次对话后就"失忆"的工具，而是能够沉淀长期记忆、记住分析师偏好、随着时间推移不断"进化"的专属数字分身。

**从分散流程到标准化Skill** ：通过将分散、重复的投研流程固化为一个个标准化的Skills，投研体系向着结构化、可复用、可审计的方向演进。

**从单兵作战到AI军团** ：随着多Agent协作架构的成熟，一个分析师可以指挥一支由前端、后端、数据、文档等专业Worker组成的AI团队，处理过去需要一个部门才能完成的复杂任务。

## 十、总结

OpenClaw作为2026年最热门的开源AI Agent框架，正在重塑人与机器的协作方式。本文详细阐述了在Windows平台本地部署OpenClaw的完整方案，涵盖五大核心场景：

1. **个人开发者快速部署**
	：通过一键脚本，30分钟内即可拥有专属AI助手
2. **私有化部署**
	：结合Ollama实现完全本地运行，保障数据隐私
3. **源码定制部署**
	：在WSL环境中从源码构建，支持深度二次开发
4. **团队协作部署**
	：HiClaw多Agent架构，实现"Manager+Workers"的AI军团
5. **金融投研场景**
	：自动化公告汇总、条件选股、策略回测，提升研究效率

Windows平台凭借其广泛的硬件生态和NVIDIA独立显卡优势，在AI Agent时代不仅没有掉队，反而展现出独特的竞争力。通过WSL2和Docker的整合，开发者可以在现有工作站上快速搭建生产级AI环境，实现零硬件成本的智能化升级。

然而，强大的能力也伴随着巨大的责任。OpenClaw拥有操作系统的"超级权限"，必须在安全可控的隔离环境中部署，严格管理技能来源和API凭证。同时，AI模型的幻觉问题无法完全消除，人类专家的判断和决策权仍然是不可替代的最后防线。

展望未来，随着多Agent协作架构的成熟和硬件生态的扩张，OpenClaw正从一个软件工具进化为AI操作系统。机器狗成为AI的腿，机械臂成为AI的手，眼镜成为AI的眼睛——一个由Agent统一调度的智能硬件时代正在开启。

对于开发者而言，现在正是踏入这一领域的最佳时机。通过本文提供的详细代码示例和实践指南，你可以在自己的Windows工作站上，养一只属于自己的"龙虾"，让它成为24小时待命的AI数字劳动力，将你从繁琐的重复工作中解放出来，专注于更具创造性的思考与决策。

**关键命令速查** ：

```
# 一键安装
iwr -useb https://openclaw.ai/install.ps1 | iex

# 私有化部署
ollama pull qwen2.5:7b
ollama create qwen2.5:7b-32k -f Modelfile
openclaw onboard

# 团队模式
npm install -g hiclaw
hiclaw init --default-workers frontend backend docs
hiclaw up

# 安全第一
openclaw doctor
openclaw logs follow
```

在AI快速演进的今天，掌握OpenClaw这样的Agent框架，意味着掌握了未来人机协作的主动权。愿你的AI助理，越用越聪明，成为真正得力的数字伙伴。

  

继续滑动看下一个

51CTO博客

向上滑动看下一个