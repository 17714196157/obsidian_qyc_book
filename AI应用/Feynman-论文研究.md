项目地址： https://github.com/companion-inc/feynman

Feynman背后跑着四个自动调度的agent：
- **Researcher**：扒论文、网页、代码库
- **Reviewer**：模拟同行评审，给出严重等级反馈
- **Writer**：从研究笔记生成结构化草稿
- **Verifier**：检查每个引用URL的存活状态，清理死链

集成的工具链也够用：
- **AlphaXiv**：论文搜索、Q&A、代码阅读
- **Hugging Face Hub**：直接读数据集元数据、schema、小文件
- **Docker**：隔离执行环境，安全跑代码
- **本地模型**：支持LM Studio、Ollama、vLLM
- **云端GPU**：Modal（突发训练）、RunPod（长时实验）

安装：
```bash
curl -fsSL https://feynman.is/install | bash

Installing Feynman 0.2.58 for Linux-x64>==> Downloading feynman-0.2.58-linux-x64.tar.gz
Extracting feynman-0.2.58-linux-x64.tar.gz==>Linking feynman into /root/.local/bin==>/root/.local/bin is already on PATH==>==> Run: hash -r && feynman
Feynman 0.2.58 installed successfully.
```
初次启动，配置大模型地址和axis登录：
```
feynman
```

| 命令                            | 描述                                               |
| ----------------------------- | ------------------------------------------------ |
| `/audit`                      | 将论文的主张与其公开代码库进行对比，识别不匹配、遗漏和可复现性风险                |
| `/autoresearch`               | 自主实验循环 — 尝试想法、测量结果、保留有效的、丢弃无效的、重复                |
| `/compare`                    | 针对某个主题比较多个来源，生成基于来源的共识、分歧和置信度矩阵                  |
| `/deepresearch`               | 深入、重来源的主题调查，生成带有内联引用的持久研究简报                      |
| `/draft`                      | 将研究发现转化为带有公式、章节和明确主张的精美论文风格草稿                    |
| `/gather-context-and-clarify` | 使用子代理收集上下文，然后提出澄清问题                              |
| `/lit`                        | 使用论文搜索和一手来源综合进行文献综述                              |
| `/parallel-cleanup`           | 并行清理审查                                           |
| `/parallel-context-build`     | 用于规划交接的并行上下文构建器                                  |
| `/parallel-handoff-plan`      | 将并行研究/上下文构建器转化为实施交接计划                            |
| `/parallel-research`          | 并行子代理研究                                          |
| `/parallel-review`            | 并行子代理审查                                          |
| `/recipe`                     | 查找由论文、数据集、文档和代码支持的排名、可实施的 ML 训练配方                |
| `/replicate`                  | 针对论文、主张或基准的复现工作流                                 |
| `/review`                     | 带有可能的反对意见、严重程度和具体修订计划的 AI 研究同行评审                 |
| `/review-loop`                | 审查/修复循环直至通过                                      |
| `/summarize`                  | 使用 RLM 模式总结任何 URL、本地文件或 PDF — 来源存储在磁盘上，从不将原始内容注入 |
