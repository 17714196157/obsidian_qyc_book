项目地址： https://github.com/OpenCut-app/OpenCut
剪映的开源平替，但是对本地电脑要求比较高，**本地可能会比较卡**
官方网页试用： https://opencut.app


# OpenCut 安装与启动指南

## 环境信息

- 系统：Windows 10
- 源码路径：`D:\code\OpenCut\OpenCut-0.3.0`
- 终端：**git-bash**（本文所有命令均以 git-bash 为准）
- 需要 Node.js 和 npm 已安装（`npm --version` 有输出即可）

---

## 一、安装 Bun

OpenCut 使用 Bun 作为包管理器，必须先安装它。

### 为什么不能直接用官方脚本？

官方安装脚本 `powershell -c "irm bun.sh/install.ps1 | iex"` 会从 GitHub 下载 Bun 二进制文件，但国内网络环境 SSL 连接 GitHub 经常失败（`schannel: failed to receive handshake`）。

npm 全局安装也会遇到权限问题——`npm install -g` 默认安装到 `D:\Program Files\nodejs\`，写入需要管理员权限，会报 `EPERM: operation not permitted`。

所以采用下面的方案：**绕过权限，安装到用户目录**。

### 安装步骤

#### 1. 修复 npm 缓存权限

npm 默认缓存目录在 `D:\Program Files\nodejs\node_cache\`，同样有权限问题。先改到用户目录：

```bash
npm config set cache "$HOME/.npm-cache"
npm cache clean --force
```

#### 2. 将 npm 全局安装目录改为用户目录

```bash
npm config set prefix "$HOME/.npm-global"
```

这样 `npm install -g` 会安装到 `C:\Users\86177\.npm-global\`，不再需要管理员权限。

#### 3. 安装 Bun

```bash
npm install -g bun
```

这一步会做两件事：
1. npm 安装 bun 的 npm 包（很小，几秒）
2. 自动执行 postinstall 脚本，从网络下载 Bun 运行时二进制文件（约 93MB）

> ⚠ **注意**：postinstall 下载二进制可能耗时较长，且 npm 可能因超时中断。如果命令卡住不动超过 2~3 分钟，按 `Ctrl+C` 退出，然后检查二进制文件是否已下载完成：

```bash
ls -lh "$HOME/.npm-global/node_modules/bun/bin/bun.exe"
```

如果文件存在（约 93MB），就说明安装成功了。如果不存在，重新运行 `npm install -g bun` 即可。

#### 4. 将 Bun 添加到 Windows PATH

**方式一：命令行（推荐，用 PowerShell 或 cmd 运行）**

```cmd
setx PATH "%PATH%;C:\Users\86177\.npm-global\node_modules\bun\bin"
```

> 如果 PATH 已经很长，`setx` 可能截断超过 1024 字符的部分。此时用方式二。

**方式二：图形界面**

1. 按 `Win + R`，输入 `sysdm.cpl`，回车
2. 点击「高级」→「环境变量」
3. 在「用户变量」中找到 `Path`，双击
4. 点击「新建」，粘贴：`C:\Users\86177\.npm-global\node_modules\bun\bin`
5. 确定保存，**重新打开终端**

#### 5. 验证安装

```bash
bun --version
```

应该输出类似 `1.3.14` 的版本号。如果提示 `command not found`，说明 PATH 没生效，检查上一步是否执行正确，然后重新打开终端。

---

## 二、安装项目依赖

### 1. 进入项目目录

```bash
cd "D:/code/OpenCut/OpenCut-0.3.0"
```

### 2. 复制环境变量文件

```bash
cp apps/web/.env.example apps/web/.env.local
```

> 如果 `cp` 命令不可用（非 git-bash 环境），用 PowerShell：
> ```powershell
> Copy-Item apps/web/.env.example apps/web/.env.local
> ```

### 3. 安装依赖

```bash
bun install
```

如果依赖下载失败，很可能是因为 npm 配置了国内镜像（如腾讯镜像 `mirrors.cloud.tencent.com`），部分大包（`next`、`@huggingface/transformers` 等）从镜像下载不稳定。改用官方源：

```bash
bun install --registry https://registry.npmjs.org/
```

#### 预期结果

安装完成后会显示类似：

```
+ turbo@2.8.20
+ typescript@6.0.2
+ @biomejs/biome@2.4.8
...
949 packages installed
```

如果看到 `Failed to install X packages`，再次运行 `bun install --registry https://registry.npmjs.org/` 补装缺失的包，直到显示 `no changes` 为止。

---

## 三、启动开发服务器

### 启动

```bash
cd "D:/code/OpenCut/OpenCut-0.3.0"
bun dev:web
```

首次启动会编译，等待约 15~20 秒，看到以下输出即成功：

```
▲ Next.js 16.1.3 (Turbopack)
- Local:         http://localhost:3000
- Network:       http://192.168.x.x:3000
✓ Ready in xx.xs
```

浏览器打开 [http://localhost:3000](http://localhost:3000) 即可访问。

### 常见问题

#### 端口已被占用

```
⚠ Port 3000 is in use by process xxxxx, using available port 3001 instead.
```

说明之前启动的进程还在。解决方法：

1. 关掉之前启动的终端窗口
2. 或在任务管理器中结束 `node.exe` / `bun.exe` 进程
3. 删除锁文件：
   ```bash
   rm -f apps/web/.next/dev/lock
   ```

#### 锁文件冲突

```
⨯ Unable to acquire lock at ...\.next\dev\lock, is another instance of next dev running?
```

删除锁文件后重试：

```bash
rm -f apps/web/.next/dev/lock
```

---

## 四、启动数据库和 Redis（可选，需要 Docker）

完整功能需要 PostgreSQL 和 Redis。如果只做前端开发，可以跳过。

```bash
cd "D:/code/OpenCut/OpenCut-0.3.0"
docker compose up -d db redis serverless-redis-http
```

首次启动会拉取镜像，可能需要几分钟。

---

## 项目结构

| 目录 | 说明 |
|------|------|
| `apps/web/` | Next.js Web 应用 |
| `apps/desktop/` | 原生桌面应用（GPUI + Rust） |
| `rust/` | 平台无关核心：GPU 合成器、特效、遮罩、WASM 绑定 |
| `docs/` | 架构和子系统文档 |

## 常用命令

```bash
bun dev:web       # 启动 Web 开发服务器
bun build:web     # 构建 Web 应用
bun lint:web      # 代码检查
bun test          # 运行测试
bun build:wasm    # 构建 WASM（编辑 rust/wasm 时需要）
```