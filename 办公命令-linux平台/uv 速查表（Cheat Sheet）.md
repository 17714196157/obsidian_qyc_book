---
title: "uv 速查表（Cheat Sheet）"
source: "https://mp.weixin.qq.com/s/i2v1Gl0-qZskXg-MDuu3wg"
author:
  - "[[liuzh]]"
published:
created: 2026-07-10
description:
tags:
  - "clippings"
---
原创 liuzh *2026年6月2日 13:01*

![[file-20260710145503049.webp]]

## uv 速查表（Cheat Sheet）

> uv — 用 Rust 编写的极速 Python 包管理器 | 官网：https://docs.astral.sh/uv

◆ ◆ ◆

## 安装 uv（不依赖 Python，独立安装）

```
# Windows（PowerShell）
irm https://astral.sh/uv/install.ps1 | iex

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证 & 升级
uv --version
uv self update
```

◆ ◆ ◆

## Python 版本管理

| 命令 | 说明 |
| --- | --- |
| `uv python install 3.11` | 安装指定版本 Python |
| `uv python install 3.11 3.12` | 同时安装多个版本 |
| `uv python list` | 列出所有已安装版本 |
| `uv python list --all-versions` | 列出所有可安装版本 |
| `uv python pin 3.11` | 固定当前项目 Python 版本（写入 `.python-version` ） |
| `uv python find` | 查看当前使用的 Python 路径 |

◆ ◆ ◆

## 虚拟环境（venv）

| 命令 | 说明 |
| --- | --- |
| `uv venv` | 创建默认 `.venv` （系统无 Python 时自动下载） |
| `uv venv --python 3.11` | 指定 Python 版本 |
| `uv venv my-env` | 自定义环境名称/路径 |
| `.venv\Scripts\Activate.ps1` | 手动激活（Windows PowerShell） |
| `source .venv/bin/activate` | 手动激活（macOS/Linux） |
| `deactivate` | 退出手动激活的环境 |
| `Remove-Item .venv -Recurse -Force` | 删除环境（Windows） |
| `rm -rf .venv` | 删除环境（macOS/Linux） |

> **关键** ： `uv run` / `uv add` / `uv pip install` 等所有 uv 命令会 **自动检测当前目录的 `.venv`** ，无需手动激活

◆ ◆ ◆

## 运行命令（自动使用.venv，无需激活）

| 命令 | 说明 |
| --- | --- |
| `uv run python main.py` | 在 venv 中运行脚本 |
| `uv run python --version` | 检查环境 Python 版本 |
| `uv run pytest` | 运行测试 |
| `uv run jupyter notebook` | 启动 Jupyter |
| `uv run -- python -c "import torch; print(torch.__version__)"` | 运行内联命令 |

◆ ◆ ◆

## 依赖管理（项目模式，推荐）

| 命令 | 说明 |
| --- | --- |
| `uv init` | 初始化项目（生成 `pyproject.toml` ） |
| `uv add requests numpy` | 添加依赖 |
| `uv add "torch==2.5.0"` | 添加指定版本 |
| `uv add torch --index-url https://download.pytorch.org/whl/cu124` | 指定源安装（GPU Torch） |
| `uv add black --dev` | 添加开发依赖 |
| `uv remove requests` | 移除依赖 |
| `uv lock` | 生成/更新锁文件 |
| `uv sync` | 按锁文件同步环境 |
| `uv sync --frozen` | 严格同步，不更新版本 |

◆ ◆ ◆

## pip 兼容模式（传统用法）

| 命令 | 说明 |
| --- | --- |
| `uv pip install requests` | 安装包 |
| `uv pip install -r requirements.txt` | 批量安装 |
| `uv pip install torch --index-url https://...` | 指定源安装 |
| `uv pip uninstall requests` | 卸载包 |
| `uv pip list` | 列出已安装的包 |
| `uv pip freeze` | 导出依赖列表（pip freeze 格式） |
| `uv pip show torch` | 查看包详情 |
| `uv pip compile requirements.in -o requirements.txt` | 锁定依赖版本 |

◆ ◆ ◆

## 全局工具管理（uv tool）

| 命令 | 说明 |
| --- | --- |
| `uv tool install black` | 安装最新版工具（隔离 venv，全局可调用） |
| `uv tool install "ruff>=0.3,<0.4"` | 安装指定版本范围 |
| `uv tool install ruff --python 3.11` | 指定工具运行的 Python 版本 |
| `uv tool list` | 列出所有已安装工具 |
| `uv tool upgrade ruff` | 升级指定工具 |
| `uv tool upgrade --all` | 升级所有工具 |
| `uv tool uninstall black` | 卸载工具 |

◆ ◆ ◆

## uvx — 临时运行工具（无需安装）

| 命令 | 说明 |
| --- | --- |
| `uvx black .` | 临时运行 black（最新版） |
| `uvx ruff@0.3.0 check .` | 临时运行指定版本 |
| `uvx ruff@latest check .` | 强制用最新版运行 |
| `uvx --from httpie http GET url` | 包名与命令名不同时用 `--from` |
| `uvx pytest` | 临时运行 pytest |
| `uvx jupyter notebook` | 临时启动 Jupyter |

> **注意** ：若已用 `uv tool install` 安装过某工具， `uvx 工具名` 默认使用已安装版本；加 `@版本号` 可临时覆盖，不影响已安装版本

◆ ◆ ◆

## 缓存管理

| 命令 | 说明 |
| --- | --- |
| `uv cache dir` | 查看缓存目录路径 |
| `uv cache list` | 列出已缓存的包 |
| `uv cache prune` | 清理过期无效缓存 |
| `uv cache clean` | 清空全部缓存（谨慎） |

**自定义缓存路径：**

```
# Windows（永久生效）
[Environment]::SetEnvironmentVariable("UV_CACHE_DIR", "D:\uv-cache", "User")

# macOS/Linux（写入 ~/.bashrc 或 ~/.zshrc）
export UV_CACHE_DIR="/data/uv-cache"
```

◆ ◆ ◆

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `UV_CACHE_DIR` | 自定义缓存目录 |
| `UV_PYTHON` | 指定默认 Python 版本 |
| `UV_INDEX_URL` | 默认 PyPI 镜像源 |
| `UV_NO_CACHE` | 禁用缓存（设为 `1` ） |
| `UV_LINK_MODE` | 链接模式（ `hardlink` / `copy` / `symlink` ） |

◆ ◆ ◆

## AI 开发 / ComfyUI 常用组合

```
# 新建 AI 项目（Python 3.11 + GPU Torch CUDA 12.4）
uv venv --python 3.11
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv run python -c "import torch; print(torch.cuda.is_available())"

# ComfyUI 替换原生 venv（省空间 80%+）
cd D:\ComfyUI
Remove-Item venv -Recurse -Force          # 删除原 venv（Windows）
uv venv venv --python 3.11               # 保留 venv 名，兼容原启动脚本
uv pip install -r requirements.txt       # 自动复用全局缓存，极速安装
```

◆ ◆ ◆

## 旧项目迁移到 uv（三步）

```
# 1. 导出旧依赖
pip freeze > requirements.txt

# 2. 删除旧环境
rm -rf .venv                              # macOS/Linux
Remove-Item .venv -Recurse -Force         # Windows

# 3. 用 uv 重建（自动复用全局缓存）
uv venv --python 3.11
uv pip install -r requirements.txt
```

◆ ◆ ◆

## 国内镜像加速

```
# 临时使用
uv pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置（pyproject.toml 或 uv.toml）
[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 其他镜像
# 阿里云: https://mirrors.aliyun.com/pypi/simple
# 腾讯云: https://mirrors.cloud.tencent.com/pypi/simple
```
