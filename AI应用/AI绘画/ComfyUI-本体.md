项目地址： 
https://github.com/comfyanonymous/ComfyUI


### windows安装
![[fc7286b39c1d9fd8bbcccb3b36b9a823_MD5.png]]
双击启动
根据显卡类型选择：
- 有 NVIDIA 显卡 → 双击 run_nvidia_gpu.bat
- 无显卡或 AMD 显卡 → 双击 run_cpu.bat
访问界面
- 启动后浏览器会自动打开 http://127.0.0.1:8188，即可开始绘图。
注意事项
- 模型需自行下载：首次使用需将 .safetensors 模型放入ComfyUI_windows_portable\ComfyUI\models\checkpoints\ 目录。
- 更新 ComfyUI：运行 update\update_comfyui.bat 即可一键更新本体。
- 插件安装：将插件解压到 ComfyUI_windows_portable\ComfyUI\custom_nodes\


### linux安装
方法1）
- 安装 Comfy CLI  pip install comfy-cli
- 安装 ComfyUI   comfy install 该命令会自动：下载 ComfyUI 主程序、安装 ComfyUI-Manager（插件管理器）
- 启动 ComfyUI   comfy launch  默认在 http://127.0.0.1:8188 启动 Web 界面。

方法2）
```
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
conda create -yn comfy  python=3.10.14
conda activate comfy
uv pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
python main.py --listen 0.0.0.0 --port 8188

可选：安装插件管理器（ComfyUI-Manager）
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ComfyUI-Manager
pip install -r requirements.txt
```
重启 ComfyUI 后，在 Web 界面中即可看到 Manager 标签页，用于安装和管理插件。
