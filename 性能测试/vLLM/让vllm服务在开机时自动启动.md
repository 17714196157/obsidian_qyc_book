---
tags:
  - vllm
  - linux部署
---

1）创建 sh脚本执行启动命令
如果需要让 vllm 服务在开机时自动启动，可以创建一个 systemd 服务。以下是具体步骤：
创建启动脚本：创建一个启动脚本，例如 /home/start_vllm.sh，
内容如下：
```python
#!/bin/bash
# 加载 conda 环境
source /home/anaconda3/etc/profile.d/conda.sh
conda activate vllm_test

CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES=1,2 \
vllm serve /home/qwq-32b-gptq-int4   \
        --served-model-name qwq-32b-gptq-int4  \
        --tensor-parallel-size 2 \
        --port 8101 \
        --max-model-len 8000 \
        --gpu-memory-utilization 0.8 \
        --uvicorn-log-level info
```
将其中的 vllm_test 替换为实际的 conda 环境名称，然后通过 chmod +x /home/username/start_vllm.sh 命令使其可执行。

2）创建 systemd 服务文件
创建 systemd 服务文件：创建  vi /etc/systemd/system/vllm.service 文件
内容如下：
```python
[Unit]
Description=VLLM Service for qwq-32b-int4-Model
After=network.target
[Service]
Type=simple
User=root
WorkingDirectory=/home
ExecStart=bash /home/vllm.sh
Restart=always
StandardOutput=journalctl
StandardError=journalctl
[Install]
WantedBy=multi-user.target
```
启动并启用服务：运行以下命令重新加载 systemd 服务，启动服务，并设置开机自启：
```python
sudo systemctl daemon-reload
sudo systemctl start vllm.service
sudo systemctl enable vllm.service
可以通过 sudo systemctl status vllm.service 命令查看服务的状态
journalctl -u vllm.service -f  查看日志
```