### 请求分布情况分析
**1\.提示词长度**
/basyGenerate/getSurgeryReportAndDailyDiseaseNoteExtract 平均prompt长度 1990.5263157894738
/getLLMBmqDiagXihua 平均prompt长度 7359.833333333333
/diag/diagXihuaByLlm 平均prompt长度 7428.642857142857
/basyGenerate/getdiagrybq 平均prompt长度 4371.867256637168
/getLLMDiseaseCureStatus 平均prompt长度 5326.428571428572
/basyGenerate/basyGenerate 平均prompt长度 3046.0

**总平均prompt长度 4539.3233830845775**

**2\.统计_时间窗口内并发量**
统计摘要:
最大并发数: 20
平均并发数: 2.36
P95并发数: 9
总病案数: 15
总接口类型: 6
峰值时刻: 2026-03-24 10:47:04
峰值并发数: 20
峰值时刻活跃病案: ['2024531804']...
峰值时刻活跃接口: ['diagXihuaByLlm', 'getdiagrybq']
结果已保存至:
  - concurrent_summary_per_second.csv (每秒汇总)
  - concurrent_details_per_second.csv (每秒Top详情)
 tips: 发现细化接口未跑完请求下，突然出现十多个入院病情的请求导致并发秃然增加
![[工作交付文档/assets/vllm现场请求耗时较长分享/7d0c57b8284893aa32c6089d52443abc_MD5.png]]

**并发特征识别**

| 指标 | 数值 | 影响 |
|------|------|------|
| **峰值并发** | 20 | 瞬间压力 |
| **平均并发** | 2.36 | 常态低负载 |
| **P95并发** | 9 | 大部分时间<9 |
| **峰谷比** | ~20:1 | 极端不均衡 |

**关键发现**：10:47:04 的峰值由 `diagXihuaByLlm` + `getdiagrybq` 两个接口叠加导致，且都是**长prompt接口**（平均7400+ 和 4300+ tokens）


**3\.对vLLM的具体影响分析**

```
突发20并发 + 长文本输入 = 三重压力:
├─ 1. Prefill阶段: 20条×~6k tokens = 120k tokens需同时处理
│   → KV Cache 预填充计算量剧增，首Token延迟(TTFT)飙升
│
├─ 2. KV Cache分配: 长文本×高并发 = 显存碎片/OOM风险
│   → 若max_model_len限制，可能触发请求排队或拒绝
│
└─ 3. 调度抖动: 从2并发→20并发，vLLM调度器来不及渐进调整
    → 可能出现preemption(抢占)，导致已生成token被踢出重算
```
结合提示词长度数据，最危险的组合：
- `diagXihuaByLlm` (7429 tokens) + `getLLMBmqDiagXihua` (7360 tokens) 同时突发
- 这两个都是诊断类接口，可能同时被同一病案触发

### 下一步猜想实验验证
- 实验1：定位瓶颈 — 分层压力测试

| 实验 | 变量控制 | 观测指标 | 目的 |
|:---|:---|:---|:---|
| **A. 固定并发，变长度** | 并发=5，prompt从1k→8k阶梯增长 | TTFT、TPOT、显存占用 | 确认长文本对prefill的影响拐点 |
| **B. 固定长度，变并发** | prompt=4k，并发从1→30阶梯增长 | 吞吐量、延迟、GPU利用率 | 找到并发 scalability 边界 |
| **C. 混合负载模拟** | 80%短请求(2k) + 20%长请求(7k)，并发渐变 | 短请求是否被长请求"饿死" | 验证调度公平性 |

- 实验2：复现现场 — 流量回放
```bash
# 基于现有CSV数据，构造真实流量模式
# 关键：保留"突发入院病情"的时间分布特征

工具建议：使用 locust/locustfile 或自定义脚本
- 常态阶段：2并发，混合接口
- 突发阶段：1秒内注入15个getdiagrybq请求
- 观测：vLLM的scheduling决策日志（--log-level=DEBUG）
```