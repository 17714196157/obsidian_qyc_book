完整代码都在 GitHub 上：https://github.com/unclecode/crawl4ai
官方文档：https://docs.crawl4ai.com/core/quickstart/

# 创建环境变量文件
```
cat > .llm.env << 'EOF'
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-d98a7434af1f4641921b8af02e175499
EOF


```

```
docker run -d \
  -p 11235:11235 \
  --name crawl4ai \
  --shm-size=2g \
  --env-file .llm.env \
  --restart=unless-stopped \
  unclecode/crawl4ai:latest
```
  
交互式 Playground：http://192.168.0.181:11235/playground — 可视化配置爬虫参数、测试抓取任务、生成 JSON 配置 
监控面板：http://192.168.0.181:11235/dashboard — 实时系统指标、浏览器池状态