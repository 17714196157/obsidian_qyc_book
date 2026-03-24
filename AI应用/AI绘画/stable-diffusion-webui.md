
```
安装stable-diffusion-webui
创建环境  conda create -n  sd python==3.10.9
https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

# 设置huggingface的镜像再下载 临时生效 ： export HF_ENDPOINT=https://hf-mirror.com

pip install uv
先执行：uv pip install -r requirements_versions.txt
再执行：uv pip install -r requirements.txt
启动：python launch.py  --port  7860 --listen --gradio-auth admin:123456 --enable-insecure-extension-access
```
![[922217665f6802e84b5d7b8c6cdfff71_MD5.png]]

如何写提示词：

![[AI应用/AI绘画/assets/stable-diffusion-webui/1c892b7e6803275f16112ce1c6554af5_MD5.png]]

![[AI应用/AI绘画/assets/stable-diffusion-webui/0f715aff7f8d3b3b012b0db33e16b3cb_MD5.png]]
提示词的权重分配

![[AI应用/AI绘画/assets/stable-diffusion-webui/d250f61fb3a3c91638102c20a0527f28_MD5.png]]

![[AI应用/AI绘画/assets/stable-diffusion-webui/9512f09588711301f9b323beddf0e21b_MD5.png]]



