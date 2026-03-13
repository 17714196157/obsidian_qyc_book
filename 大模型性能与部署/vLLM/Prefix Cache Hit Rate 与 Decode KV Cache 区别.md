
| 指标                        | 含义                                                         | 默认值     | 使用阶段                   | 细节区分                                                                                                                  |
| ------------------------- | ---------------------------------------------------------- | ------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Prefix Cache Hit Rate** | 跨请求复用比例（APC）让请求N **复用** 请求1~N-1的缓存， **跳过** 请求N的部分Prefill计算 | 有（日志打印） | Prefill阶段,**系统级跨请求指标** | 物理存储的 KV Cache：<br>   - 包含位置编码信息（ROPE/alibi等）<br>   - 是绝对的、不可复用的（如果位置不同）`[A,B]` 和 `[B,A]` 的注意力输出不同，因为位置编码不同           |
| **Decode KV Cache 命中率**   | 实际是 **存储的键值对张量**                                           | 无（不打印）  | Decode 阶段，内存访问层面       | APC 的缓存索引：<br>   - 只记录 "这个 token 序列曾经计算过"<br>   - 命中后，**重新应用当前位置的位置编码**<br>   - 或者：vLLM 实际上存储的是 **未加位置编码的 KV**，应用时再编码 |

```markdown
# 日志示例
Engine 000: Running: 39 reqs, Waiting: 0 reqs, 
            GPU KV cache usage: 68.9%, 
            Prefix cache hit rate: 29.7%
```

### 一）APC（Automatic Prefix Caching)
**APC定义**：
 - 查询单位 ：每次请求到来时，会查询其 prompt tokens 能命中多少个已缓存的 block
- 统计方式 ：记录 `vllm:prefix_cache_queries` （查询次数）和 `vllm:prefix_cache_hits` （命中token数）
- 日志显示 ： **最近1000次查询的滑动窗口平均命中率**
- 核心机制：块级（Block-level）前缀匹配
	- a) 结构化Prompt：System Prompt单独成块，User Query新起一块
	- b) 位置无关性：System Prompt单独成块，User Query新起一块
##### 困惑问题
- 1：命中率 100% 但 TTFT 不为零（命中率 100% ≠ TTFT 为 0）？即使 100% 命中，仍有以下开销：

| 开销项             | 说明                          |
| --------------- | --------------------------- |
| **缓存查找**        | Radix Tree 遍历，O(长度)         |
| **内存拷贝**        | 从 CPU 管理的缓存拷贝到 GPU KV Cache |
| **位置编码重计算**     | 如果存储的是 base KV，需要重新应用 ROPE  |
| **后续 token 计算** | 只有前缀免计算，新 token 仍需 Prefill  |
-2 : 为什么命中率会波动，甚至之前命中的后来不命中了？
```python
GPU 显存有限 → KV Cache 容量有限 → 需要驱逐

场景：
1. 请求A：长文档X（占用 100 blocks）
2. 请求B：长文档Y（占用 100 blocks）→ 可能驱逐文档X的部分块
3. 请求C：文档X（之前能 100% 命中，现在可能 0%）

LRU 策略：最近最少使用的块被驱逐
1. 缓存是 **全局共享** 的，不是按请求隔离
2. 高并发时，热点内容可能互相驱逐
3. 长文本更容易导致缓存抖动

```

### 二）KV Cache
理论上是 100% "使用"，但不是"命中"，Decode 阶段的 KV Cache 是 **确定性访问** ，不是 **缓存查找** 
```markdown
Decode 步骤 5：
- 需要读取：Token 0-4 的 KV（之前生成的）
- 需要计算：Token 5 的 KV（新生成的）

内存访问模式：
- GPU 从 HBM 读取 Token 0-4 的 KV → 这是"访问"，不是"命中/未命中"
- 没有"未命中"概念，因为这些 KV 一定在显存里（刚算出来的）
```
但存在以下 **性能问题** ：

| 问题             | 现象                | 原因                   |
| -------------- | ----------------- | -------------------- |
| **Page Fault** | 需要分配新 block       | 显存碎片或不足              |
| **重计算**        | 重新计算之前 token 的 KV | 显存不足导致早期 KV 被驱逐（极少见） |
| **CPU-GPU 拷贝** | 从 CPU 缓存加载 KV     | 使用 CPU offloading 时  |

Decode 阶段真正瓶颈的问题 **不是命中率** ，而是 **内存带宽瓶颈** Memory-Bound：
```markdown
Decode 每步计算量：O(1) （只算一个新 token）
Decode 每步内存访问：O(N) （读取之前所有 N 个 token 的 KV）

当 N > 1000 时，95% 时间花在读取 KV Cache 上
```
这就是为什么需要：
- **KV Cache 量化** （INT8/FP8）：减少内存读取量
- **PagedAttention** ：减少内存碎片，提高连续访问效率
- **FlashAttention-Decode** ：优化内存访问模式

##### 观察到 **Decode 阶段很慢** ，可能的原因：
1. **序列太长** → KV Cache 读取量线性增长 → 使用 KV Cache 量化
2. **Batch size 太小** → GPU 计算单元空闲 → 增大 batch size 或用 continuous batching
3. **显存碎片** → PagedAttention 应该已解决，但极端情况仍存在
4. **CPU-GPU 数据传输** → 如果用了 CPU offloading，检查 `nvidia-smi` 的 PCIe 带宽


##### vLLM 默认不打印，如果观察kv cache效率
- 方法 1：启用详细日志
```bash
# 启动时增加
export VLLM_LOGGING_LEVEL=DEBUG
# 或
python -m vllm.entrypoints.openai.api_server --log-level DEBUG
```
- 方法2：使用 Prometheus 指标（如果有部署）
```python
# vLLM 暴露的 metrics 中可能有
vllm:time_to_first_token_seconds  # Prefill 时间
vllm:time_per_output_token_seconds  # Decode 每 token 时间
```


### 三）**RadixAttention** 算法的核心特性：

| 特性         | 说明                                           |
| ---------- | -------------------------------------------- |
| **内容寻址**   | Cache 以 **token 序列的哈希值** 为 key，而非 (请求ID, 位置) |
| **树状结构**   | 所有请求的 KV Cache 组成一棵 **Radix Tree（基数树）**      |
| **最长前缀匹配** | 新请求从根节点开始，沿树匹配最长的相同 token 序列                 |

```python
Radix Tree 结构示例：

Root
├── "You are a helpful assistant." [Block 0]
│   ├── "User: Hello" [Block 1]
│   │   └── "Assistant: Hi" [Block 2]
│   └── "User: How are you" [Block 1']
│       └── "Assistant: I'm fine" [Block 2']
│
└── "You are a coding expert." [Block 0']
    └── ...

请求C: "You are a helpful assistant. User: Hello..."
       从 Root → Block 0 → Block 1 完全命中！
       不需要知道 Block 0 是"第0个块"，只需要内容匹配
       
# 具体实现：Block 的哈希计算
# vLLM 简化逻辑（基于源码理解）
class Block:
    def compute_hash(self):
        # 只哈希当前块内的 token IDs
        # 不包含：请求ID、块位置、时间戳
        return hash(tuple(self.token_ids))
    
    def matches(self, other_block):
        return self.hash == other_block.hash
        # 位置不重要，内容相同即可
```
- 匹配必须 **从根节点开始连续**
- 一旦某个块不匹配，后续即使内容相同也无法命中
- 这是 **最长前缀匹配** 的特性，不是模糊匹配

**优化建议：**
- 固定模板用 **无意义填充** 对齐块边界
- 动态内容 **单独成块** ，不要和固定内容混在一个块里
