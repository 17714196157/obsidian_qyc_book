---
tags:
  - obsidian操作知识
---

obsidian插件下载异常 自动使用代理插件 GitHub仓库：https://github.com/xuha233/obsidian-market-proxy/tree/feature/proxy-fix-and-enhancements

插件列表地址：https://airtable.com/appErQxa3n8SnyUdO/shrdmp10Lxmf5Wmgl/tblJqnWpcKURTjysX

tips：
obsidian插件是安装在库更目录下的 obsidian_qyc_book\.obsidian\plugins 里，所以换库会导致插件无法同步过去

### 推荐obsidian插件
##### 1） 资源管理
- Custom Attachment Location 插件 ---> 插入图片时会创建一个和笔记名字一样的文件夹用于存放附件资源 assets\长文本处理场景处理方案讨论
- Clear Unused Images 插件  ---> 删除未使用的图片资源
- Local Images Plus 插件---> 它会自动将笔记中的外部图片链接下载到本地
##### 2)markdown格式辅助
- Linter   插件 --->  格式化空格 空行
- Mind Map  插件 --->  markdown转思维导图
- Mermaid Tools 插件  ---> 辅助化Mermaid图,可以给一个框架模板

##### 3）模板
- Templater 插件 --->  创建笔记模板， 命令面板（Ctrl + P） 选择 Templater: Open insert template modal 命令， 然后选择已经创建好的模板插入到笔记中。

### 微信公众号文章同步方法
##### 1.开源浏览器插件：文章下载器（纯本地，最安全）
- GitHub: [markdown-export](https://github.com/yang-shuohao/markdown-export)
- 优点：
    - 纯前端实现，**无需登录，无需后端**，数据完全本地处理
    - 支持知乎、CSDN、**微信公众号**、博客园
    - 自动提取标题、正文、图片、代码块转为标准 Markdown（图片会是网络链接）
- 安装：Chrome 开发者模式加载已解压的扩展程序即可
- 操作方法：Chrome打开微信公众号文章，右击选择《导出文章为markdown》
##### 2.Obsidian Web Clipper 浏览器插件
- **优点**：无需第三方服务，直接通过浏览器插件剪藏到 Obsidian
- GitHub:[obsidianmd/obsidian-clipper](https://github.com/obsidianmd/obsidian-clipper/)
- **配置**：
    1. 浏览器安装 **Obsidian Web Clipper** 插件
    2. 浏览器插件配置  导出obsidian的仓库目录
![[ObsidianWebClipper浏览器插件配置1.jpg]]