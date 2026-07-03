## 常用命令

```bash
# 查看所有 容器状态 
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.State}}"  | grep vllm

```