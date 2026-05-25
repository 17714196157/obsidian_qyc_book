
**一句话介绍**：输入文案或关键词，自动生成带配音、字幕、BGM 的完整短视频，支持批量生产。
   核心功能
-  **一键生成短视频**：输入文案或关键词，自动完成配音、字幕、BGM 组合
-  **多种视频比例**：支持 9:16（短视频）、16:9（横屏）、1:1（方形）
-  **丰富的视频素材**：集成 Pexels 免费素材库，海量高清视频片段
- **AI 配音**：支持多种语音引擎，包括 Microsoft Azure、ElevenLabs 等
- **字幕自动生成**：自动识别语音生成字幕，支持多种样式
- **Web 可视化界面**：无需代码，浏览器即可操作

 **安装：** 
```
git clone https://github.com/harry0703/MoneyPrinterTurbo.git  
cd MoneyPrinterTurbo

创建虚拟环境并安装依赖
conda create -n MoneyPrinterTurbo python=3.11
conda activate MoneyPrinterTurbo
pip install -r requirements.txt

安装 FFmpeg
sudo apt-get install ffmpeg


配置文件
复制示例配置：
cp config.example.toml config.toml
配置 API Key
编辑 config.toml：
[app]
# Pexels API Key（免费获取）
pexels_api_keys = ["your_pexels_api_key_here"]

# OpenAI API Key（可选，用于文案生成）
openai_api_key = "your_openai_api_key"

# Azure 语音配置（可选，用于 AI 配音）
azure_speech_key = "your_azure_speech_key"
azure_speech_region = "eastasia"
获取 Pexels API Key：

1. 访问 https://www.pexels.com/api/
2. 注册账号并申请 API Key
3. 免费版每小时 200 次请求

启动方式
Web 界面（推荐）
python webui.py
```

![[自媒体工具/视频制作/assets/MoneyPrinterTurbo/540a75568d106018c8a8e5f5b8e7c7ed_MD5.png]]