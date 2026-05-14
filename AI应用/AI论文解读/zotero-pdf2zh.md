---
title: Zotero PDF2ZH 论文翻译插件
source: https://github.com/guaguastandup/zotero-pdf2zh
created: 2026-05-14
tags:
  - zotero
  - paper/translation
  - pdf2zh
  - tools
---

# Zotero PDF2ZH 论文翻译插件

> 通过 [[zotero-pdf2zh]] 插件实现 Zotero 内论文 PDF 的 AI 翻译，支持双语对照和纯译文输出。

**项目地址**：https://github.com/guaguastandup/zotero-pdf2zh

---

## 第一步：创建目录结构

```bash
# 1. 创建并进入工作目录
mkdir zotero-pdf2zh && cd zotero-pdf2zh

# 2. 下载 server.py 文件
wget https://raw.githubusercontent.com/guaguastandup/zotero-pdf2zh/refs/heads/main/server.py

# 3. 创建 translated 文件夹，存放翻译输出文件
mkdir translated

# 4. 创建 config.json 文件（待配置）
echo '{}' > config.json
```

最终文件夹结构：

```
zotero-pdf2zh/
├── config.json
├── server.py
└── translated/
```

---

## 第二步：安装依赖

```bash
# 创建 conda 虚拟环境
conda create -n zotero-pdf2zh python=3.12

# 激活环境
conda activate zotero-pdf2zh

# 安装核心依赖
pip install pdf2zh==1.9.6 flask pypdf

# 修正 pdfminer 版本（避免兼容问题）
pip install pdfminer.six==20250416

# 固定 numpy 版本
pip install numpy==2.2.0
```

> [!warning] 版本兼容性
> 必须锁定 `pdfminer.six==20250416` 和 `numpy==2.2.0`，否则可能出现兼容问题。

---

## 第三步：配置文件

### config.json

> [!danger] 安全提醒
> 配置文件中包含 API Key，请勿将文件上传到公开仓库或分享给他人。

```json
{
    "USE_MODELSCOPE": "0",
    "PDF2ZH_LANG_FROM": "English",
    "PDF2ZH_LANG_TO": "Simplified Chinese",
    "NOTO_FONT_PATH": "./LXGWWenKai-Regular.ttf",
    "translators": [
        {
            "name": "zhipu",
            "envs": {
                "ZHIPU_API_KEY": "your_zhipu_api_key",
                "ZHIPU_MODEL": "glm-4-flash"
            }
        },
        {
            "name": "openailiked",
            "envs": {
                "OPENAILIKED_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "OPENAILIKED_API_KEY": "your_dashscope_api_key",
                "OPENAILIKED_MODEL": "deepseek-v3"
            }
        }
    ]
}
```

### 可用翻译服务

| 服务名 | 环境变量 | 说明 |
|--------|---------|------|
| `zhipu` | `ZHIPU_API_KEY`、`ZHIPU_MODEL` | 智谱 AI |
| `openailiked` | `OPENAILIKED_BASE_URL`、`OPENAILIKED_API_KEY`、`OPENAILIKED_MODEL` | 兼容 OpenAI 接口的服务（如阿里 DashScope） |

---

## 第四步：测试翻译

### 设置环境变量

```bash
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="your_dashscope_api_key"
export OPENAI_MODEL="deepseek-v3"

export OPENAILIKED_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAILIKED_API_KEY="your_dashscope_api_key"
export OPENAILIKED_MODEL="deepseek-v3"
```

### 执行翻译

```bash
pdf2zh 2506.14755v1.pdf --service openailiked --debug
```

### 输出说明

| 文件类型 | 说明 |
|---------|------|
| **dual** | 包含源语言和目标语言（双语对照） |
| **mono** | 仅包含目标语言（纯译文） |

![[AI应用/AI论文解读/assets/zotero-pdf2zh/f7571b28a8d2226cb3b1ae3d793dd8c5_MD5.png]]

---

## 第五步：对接 Zotero

### 1. 启动后端服务

```bash
python server.py 8888
```

### 2. 安装 Zotero 插件

> [!info] 版本要求
> 必须使用 **Zotero 7** 及以上版本。

1. 下载插件：https://zotero-chinese.com/plugins/#search=pdf2
2. 打开 Zotero → **工具** → **插件** → 右上角 **Install Plugin From File…**
3. 选择下载的插件文件

![[file-20260514151738803.png]]

### 3. 使用翻译

选择 PDF → 点击翻译 → 发送给后台 `server.py` 端口 → 翻译完成后返回翻译版 PDF，Zotero 界面即可看到翻译结果。

![[c0c641ee1059459af76626177d081ebe_MD5.png]]

---

## 方法二：PDF2ZH Next（GUI 版本）

> 与之前的库安装存在兼容问题，可作为替代方案。

**项目地址**：
- https://github.com/PDFMathTranslate/PDFMathTranslate-next
- https://github.com/guaguastandup/zotero-pdf2zh/blob/main/README_babeldoc.md

### 安装

```bash
uv pip install pdf2zh_next pypdf flask
```

### 启动 GUI

```bash
pdf2zh_next --gui
```

进入图形界面即可拖拽文件进行翻译。

![[AI应用/AI论文解读/assets/zotero-pdf2zh/cb54f2e820c5307cbac9148f206ffe8d_MD5.png]]
