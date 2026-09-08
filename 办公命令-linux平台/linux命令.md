```bash
# 查看进程完整命令行，确定显卡进程谁在
ps -fp 1830105,18908,18997,14461,14462,14485

# 查看进程的工作目录（判断是哪个项目）
ls -l /proc/18908/cwd

```

**查询显卡，那些进程在占用**
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv && echo "---" && ps -fp $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
```