### 安装
##### 安装前准备
```
# 1. 先安装 Node.js（如未安装）
访问 https://nodejs.org/ 下载 LTS 版本并安装

# 2. 验证 Node.js 安装
node -v
npm -v

# 3. 安装 nvm（Node 版本管理器）
国内网络加速版（使用镜像）
curl -o- https://gitee.com/mirrors/nvm/raw/master/install.sh | bash
### 加载 nvm 环境 
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
nvm install 22 & nvm use 22    # 安装 Node 22 & 切换到 Node 22
nvm alias default 22 # 设为默认版本

# 4. cmake版本升级
sudo apt remove cmake
sudo snap install cmake --classic

```

tips： 管理员权限打开cmd
配置介绍开源工具切换大模型源地址：**CCSwitch**（地址：[cc-switchv3.11.1 · farion1231/cc-switch · GitHub](https://github.com/farion1231/cc-switch/releases/tag/v3.11.1)）

##### 安装opencode
方式 1：NPM 安装
opencode安装：npm install -g opencode-ai@latest
```

opencode 官方文档： [doc_opencode](http://opencode.ai/docs/zh-cn/config)

##### opencode选择模型：
参考文档 [providers-自定义提供商](http://opencode.ai/docs/zh-cn/providers/#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8F%90%E4%BE%9B%E5%95%86)
修改配置文件，增加自定义的供应商
```
cat ~/.config/opencode/opencode.json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "myprovider": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "阿里云百炼",
      "options": {
          "baseURL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
           "apiKey": "",
      },
      "models": {
        "deepseek-v3": {
          "name": "deepseek-v3",
          "limit": {
            "context": 64000,
            "output": 8192
          }
        },
		 "qwen2.5-14b-instruct": {
          "name": "qwen2.5-14b-instruct",
          "limit": {
            "context": 18000,
            "output": 8192
          }
        },
        "qwq-32b": {
          "name": "qwq-32b",
          "limit": {
            "context": 32000,
            "output": 8192
          }
        }
        }

    }
  }
}
```
opencode命令窗口  /connect 选择自己注册的阿里云百炼
opencode命令窗口  /models 选择自己的模型
```

### opencode的 oh-my-opencode 插件安装：
```
npx oh-my-opencode install
# 检查插件是否加载 ,确保 `plugin` 数组里包含 `oh-my-opencode`
cat ~/.config/opencode/opencode.json | grep -A5 '"plugin"'
```
oh-my-opencode 介绍 [[Oh-My-OpenCode 完全指南]]


### opencode命令
OPENCODE_ENABLE_EXA=1 opencode   # 启动服务，打开在网络上搜索信息