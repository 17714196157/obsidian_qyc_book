## 更新驱动
```
wget --referer="https://www.nvidia.com/" \
  --user-agent="Mozilla/5.0" \
  https://us.download.nvidia.com/tesla/580.126.20/NVIDIA-Linux-x86_64-580.126.20.run
  

# 停止可能占用 GPU 的服务
sudo systemctl stop docker nvidia-persistenced 2>/dev/null || true

# 杀掉所有使用 GPU 的 Python/CUDA 进程
sudo fuser -v /dev/nvidia* 2>/dev/null
sudo killall -9 python python3 ft 2>/dev/null || true

# 退出 FreeToken 虚拟环境（当前 shell 里）
deactivate 2>/dev/null || true

卸载旧驱动模块
bash
# 按顺序移除内核模块（必须先移除依赖项）
sudo rmmod nvidia_uvm 2>/dev/null || true
sudo rmmod nvidia_drm 2>/dev/null || true
sudo rmmod nvidia_modeset 2>/dev/null || true
sudo rmmod nvidia 2>/dev/null || true

# 验证是否全部卸载
lsmod | grep nvidia


# 重新安装（这次不需要 --no-questions，用交互式更安全）
sudo ./NVIDIA-Linux-x86_64-580.126.20.run \
  --no-x-check \
  --no-nouveau-check \
  --no-opengl-files
  
必须执行 sudo reboot 重启
重启后重新连接，执行 nvidia-smi 验证驱动版本
root@maizi:~# nvidia-smi
Fri Sep  4 18:22:19 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.20             Driver Version: 580.126.20     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:5E:00.0 Off |                    0 |
| N/A   52C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```



ft serve \
  --model /home/qyc/bert/Qwen3.8-27B \
  --moe-backend offload \
  --memory-ratio 0.7

git clone https://github.com/FlashML-org/FreeToken
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
ft serve  --model /home/qyc/bert/Qwen3.8-27B --host 0.0.0.0 --port 1919  --moe-backend offload --memory-ratio 0.9 