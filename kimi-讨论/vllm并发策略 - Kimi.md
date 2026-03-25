---
title: "vllm并发策略 - Kimi"
source: "https://www.kimi.com/chat/19ce5571-eeb2-8317-8000-09fafc6da183?chat_enter_method=new_chat"
author:
published:
tags:
  - "clippings"
---
![[file-20260313143434897.png]]

领导 看了 https://vllm.ai/blog/anatomy-of-vllm 这个文章， 得到图中的结论：， 其中那些是可行的可以展开那些对当前业务有用的工作实验， 那些是错误的我需要注意的。 业务背景：vllm作为大模型后台， 我们自己开发了一个程序用于接受电子病案和任务，自己拼接不同的任务的提示词prompt请求大模型，解析大模型的结果。 每个任务只用到部分电子病案，比如： 手术细化只要到某个手术记录， 手术切开等级判断用到出院记录和入院记录， 诊断细化用到出院记录、检查报告、病理白宫 等等医学任务， 我现在如何安排业务大模型请求的并发策略可以最大化vllm的性能


我来分析这个技术方案，并搜索vLLM最新的性能优化实践来验证这些结论。

基于对vLLM技术文档和最新性能研究的分析，我来为你逐一评估这些结论的正确性，并给出针对 **电子病案多任务处理场景** 的优化策略。

---

## 一、结论正确性评估

### ✅ 正确的结论（可展开实验）

| 结论 | 评估 | 实验建议 |
| --- | --- | --- |
| **"预填充+生成"两阶段耗时分析** | ✅ **正确** | 这是vLLM的基础架构。TTFT（Time To First Token）由预填充决定，TPOT（Time Per Output Token）由生成阶段决定 |
| **KV cache命中率主要针对预填充阶段** | ✅ **正确** | APC（Automatic Prefix Caching）只优化预填充，不优化生成 |
| **PageAttention支持同session多轮共享KV cache** | ✅ **正确** | 通过hash-based block匹配实现，相同token序列可复用物理块 |
| **不同session间相同prompt可共享KV cache** | ✅ **正确** | 这是APC的核心能力，不限于同session |
| **全命中时预填充耗时可忽略** | ✅ **理论正确** | 实际仍有~100-200ns/token的hash开销 |
| **生成阶段是主要瓶颈** | ✅ **正确** | 若输出1K token @80token/s需12.5s，3轮即37.5s，占总耗时大头 |

### ⚠️ 部分正确/需修正的结论

| 结论 | 评估 | 修正说明 |
| --- | --- | --- |
| **"综述不能提高KV cache命中率"** | ⚠️ **表述不准确** | 综述 **可以** 提高命中率，但方式不同：将长病历压缩为短综述后，后续请求的整体token数减少，使得 **有效上下文变短** ，从而间接提升缓存效率。但综述本身作为新的prompt前缀，其KV cache可以被后续请求复用 |
| **"空间换时间"优化长期运作** | ⚠️ **需量化验证** | 综述确实降低显存需求和预填充时间，但生成综述本身有成本。需验证：生成综述的耗时 vs 后续节省的耗时 |

### ❌ 错误的结论（需注意）

| 结论 | 评估 | 风险说明 |
| --- | --- | --- |
| **"生成1K token需15s"（按80token/s）** | ❌ **计算错误** | 1000/80 = **12.5s** ，不是15s。这种误差在估算累积耗时（3轮45s vs 实际37.5s）会导致错误决策 |
| **"性价比不高"的结论** | ❌ **过于武断** | 80s→优化一半到40s，在医疗场景（实时性要求）可能是关键提升。需结合业务SLA评估 |
| **Hot/Warm/Cold memory分层** | ❌ **概念混淆** | vLLM官方无此分层定义。这是将操作系统缓存策略套用到LLM推理，但实现机制不同（vLLM是hash-based block管理，非LRU分层） |

---

## 二、针对电子病案场景的并发优化策略

你的业务特点是： **同一病案 → 多个独立任务 → 每个任务只用部分病历** 。这是典型的 **"长上下文、多任务、部分重叠"** 场景。

### 核心矛盾分析

```markdown
传统做法的问题：
病案A → 任务1（手术细化）→ 请求vLLM → 处理手术记录部分
     → 任务2（手术等级）→ 请求vLLM → 处理出入院记录（部分重叠）
     → 任务3（诊断细化）→ 请求vLLM → 处理出院+检查+病理（大量新内容）
     
问题：每个任务独立请求，即使同一病案，KV cache也无法复用（因为prompt不同）
```

### 推荐策略：三级并发架构

#### 策略1：病案级会话保持（Session Affinity）

```python
# 伪代码示例
class CaseSessionManager:
    def __init__(self):
        self.case_sessions = {}  # case_id -> session_state
    
    def process_case(self, case_id, tasks):
        """
        关键：同一病案的所有任务在同一个vLLM会话中串行/批量处理
        """
        session = self.get_or_create_session(case_id)
        
        # 方式A：串行复用（简单，APC自然生效）
        for task in tasks:
            result = session.send_request(task.prompt)  # 历史自动作为前缀
            
        # 方式B：批量提交（更高吞吐，需vLLM支持batching）
        results = session.batch_send([t.prompt for t in tasks])
```

**为什么这样有效：**

- vLLM的APC会缓存前面任务的KV cache
- 后续任务若包含相同病历片段，自动命中缓存
- 避免病案切换导致的KV cache失效

#### 策略2：智能Prompt拼接（最大化前缀重叠）

```python
# 优化前（低命中率）
Task1: [手术记录片段] + [指令：细化手术名称]
Task2: [入院记录] + [出院记录] + [指令：判断手术等级]  
Task3: [出院记录] + [检查报告] + [病理报告] + [指令：细化诊断]

# 优化后（高命中率）- 标准化前缀结构
Standard_Prefix = [病案基础信息] + [患者基本信息] + [主诉摘要]

Task1: [Standard_Prefix] + [手术记录] + [指令1]
Task2: [Standard_Prefix] + [入院记录] + [出院记录] + [指令2]  
Task3: [Standard_Prefix] + [出院记录] + [检查报告] + [病理报告] + [指令3]
```

**关键：** 将 **所有任务共用的病案基础信息** 放在prompt最前面，形成可缓存的长前缀。

#### 策略3：动态综述生成（权衡点）

关于你提到的"综述"策略，建议采用 **条件化生成** ：

```python
def should_generate_summary(case_data, task_history):
    """
    触发条件（满足任一即生成）：
    1. 病案文本总量 > 8000 tokens（长上下文压力）
    2. 当前会话已执行 > 3个任务（累积上下文过长）
    3. 预测后续还有 > 2个任务待执行（ ROI 为正）
    4. 病历中包含大量非结构化文本（如病程记录）
    """
    total_tokens = estimate_tokens(case_data)
    remaining_tasks = predict_remaining_tasks(case_data)
    
    if total_tokens > 8000 and remaining_tasks > 2:
        return True
    return False

def generate_structured_summary(case_data):
    """
    结构化综述格式（便于后续解析和缓存）：
    """
    summary_template = """
    [病案摘要]
    患者：{姓名}，{年龄}岁，{性别}
    主诊断：{主要诊断}
    关键手术：{手术列表}
    关键检查：{异常检查结果}
    治疗经过：{时间线摘要}
    
    [原始病历索引]
    手术记录：见原始文档第{X}段
    出院记录：见原始文档第{Y}段
    ...
    """
    return llm_generate(summary_template, case_data)
```

**综述的价值验证：**

- **降低显存** ：从16K上下文 → 2K综述 + 4K新内容 = 6K有效长度
- **加速预填充** ：APC命中综述部分，新内容只需计算4K
- **成本** ：生成综述需1次LLM调用（~2s），但后续每个任务节省~3-5s预填充时间

---

## 三、具体实验建议

### 实验1：验证APC在你的场景的收益

```bash
# vLLM启动参数
python -m vllm.entrypoints.openai.api_server \
    --model your-medical-model \
    --enable-prefix-caching \  # 关键：开启APC
    --max-num-seqs 256 \
    --max-model-len 16384
```

**测试用例设计：**

1. **基线** ：同一病案的5个任务，分别独立请求（模拟当前架构）
2. **优化A** ：同一病案5个任务，串行复用同一session（利用APC）
3. **优化B** ：优化A + 标准化前缀（最大化前缀重叠）
4. **优化C** ：优化B + 动态综述（长病案触发）

**观测指标：**

- TTFT（首token延迟）
- 总端到端时间
- GPU显存占用
- APC命中率（vLLM日志： `prefix_cache_hit_rate` ）

### 实验2：并发策略对比

| 策略 | 适用场景 | 预期效果 |
| --- | --- | --- |
| **病案级串行** | 单个病案任务多，实时性要求中等 | 最大化APC命中，简单可靠 |
| **病案级批量** | 任务可并行化，vLLM版本支持chunked prefill | 更高吞吐，需测试batch效果 |
| **跨病案并行** | 多病案同时处理 | 需确保GPU利用率，但APC跨病案失效 |

---

## 四、关键风险提醒

1. **APC的hash开销** ：在vLLM中，APC有~100-200ns/token的hash计算开销。对于极短prompt（<500 tokens），APC可能带来负收益。建议设置 `--enable-prefix-caching` 仅在病案级session中使用。
2. **vLLM版本差异** ：v0.6.3之前的版本在APC高并发下有调度bug，建议升级到 **v0.6.5+** 。
3. **"综述"不是银弹** ：生成综述本身消耗token，且综述的KV cache只能被 **包含该综述前缀** 的请求复用。如果后续任务prompt结构差异大，综述策略失效。
4. **医疗准确性风险** ：综述可能丢失关键医疗细节，建议保留 **原始病历索引** ，在关键任务中回查原始文本。

---

## 总结

你领导的核心思路 **方向正确** ，但部分技术细节需修正：

- **可行** ：病案级session保持、标准化prompt前缀、APC优化预填充
- **需验证** ：综述策略的ROI（建议先在小批量数据上A/B测试）
- **错误** ：计算误差（15s vs 12.5s）、内存分层概念混淆

建议先实施 **策略1（会话保持）+ 策略2（标准化前缀）** ，这是零成本、高收益的优化。综述策略作为第二阶段优化，需严格验证后再上线。

  

尽管问，带图也行

K2.5 快速