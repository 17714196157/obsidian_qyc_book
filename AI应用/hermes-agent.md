官网文档 https://hermes-agent.nousresearch.com/docs/getting-started
安装：
```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc

hermes model # Choose your LLM provider and model  
hermes tools # Configure which tools are enabled  
hermes setup # Or configure everything at once
hermes  # 启动交互应用界面

hermes chat -q "测试"

```
**配置文件路径： Hermes Agent 的主配置文件 `~/.hermes/config.yaml`**

启动UI管理界面 
```bash
hermes dashboard --host 0.0.0.0 --port 1111 --insecure
```
![[Hermes-AgentUI界面.png]]