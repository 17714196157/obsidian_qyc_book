
![[测试并发脚本日志.png]]

第一个是按tokens计算， 第二个按字符串长度计算。
![[vllm日志.png]]


### 测试vllm的并发脚本
```python
import json
import concurrent.futures
import requests
import time

api_url = "http://192.168.0.172:8000/v1/completions"   # 修改为你的 vLLM 地址
# api_url = "http://localhost:8102/v1/completions"
prompt = "Once upon a time,"


max_tokens = 2000
model = "/home/qyc/bert/Qwen2-0.5B" 
# model =  "qwq-32b"  # "/home/qyc/bert/Qwen2-0.5B"  # "qwq-32b"

import time
import requests


def post_http_request(api_url: str, pload, request_id: int) -> dict:
    t1 = time.time()
    headers = {"User-Agent": "Test Client"}
    try:
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
        data = json.loads(response.content)
        # output = data.get("text", [""])[0] if isinstance(data.get("text"), list) else data.get("text", "")
        output = data.get("choices", [{'text': ''}])[0]["text"]
        # request_id = data.get("id", "000")
        len_output = data.get('usage',{})["completion_tokens"]
        t2 = time.time()
        costtime = t2-t1
        print(f"request_id={request_id} costtime={costtime} {len_output/costtime}token/s usage{data.get('usage',{})}")
        
        
        return {"id": request_id, "output": output, "params": pload}
    except Exception as e:
        return {"id": request_id, "error": str(e)}

		
def test_single_vs_concurrent():
    """
    对比单请求 greedy 和并发 greedy 的结果
    """
    # api_url = "http://192.168.0.172:8102/v1/completions"  # 修改为你的 vLLM 地址
    # prompt = "Once upon a time,"
    
    print("\n" + "=" * 60)
    print("对比测试: 单请求 vs 并发请求")
    print("=" * 60)
    
    # 先单独发一个 greedy 请求，记录基准结果
    print("\n1. 发送单请求获取基准结果...")
    baseline_payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "model": model,
        "seed": 42  ,   
        # 关键：禁用并行推理中的非确定性优化
        # "use_beam_search": False,  # 确保贪婪解码
        # "enforce_eager": True,  
        "frequency_penalty": 0.0,     # 禁用惩罚
        "presence_penalty": 0.0,
        # "stop": ["\n", "。", "}"],     # 强制停止词，控制生成长度
        # "ignore_eos": False,          # 必须允许 EOS
        "max_tokens": max_tokens
    }
    baseline_list = []
    for i in range(3):
        baseline = post_http_request(api_url, baseline_payload, request_id=i)
        baseline =  baseline["output"]
        # response = requests.post(api_url, headers={"User-Agent": "Test"}, json=baseline_payload, stream=True)
        # data = json.loads(response.content)
        # baseline = data.get("choices", [{'text': ''}])[0]["text"]
        # print(f"{i}  基准结果: {data.get('usage',{})}")

        baseline_list.append(baseline)
    print("\n1.1  串行3个greedy请求...")
    print(f" 检测到串行3个greedy请求 {len(set(baseline_list))} 种不同的输出：")
    print(f"baseline={baseline}")

    # 然后并发发多个 greedy 请求
    max_workers = 5
    print(f"\n2. 并发发送 {max_workers} 个 greedy 请求...")
    time.sleep(1)  # 等待一下
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(20):
            futures.append(
                executor.submit(post_http_request, api_url=api_url, pload=baseline_payload, request_id=i)
            )
        
        concurrent_results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            concurrent_results.append(result.get("output", ""))
            # print(f"   并发结果 {result['id']}: {len(result.get('output', ''))} {result.get('output', '')[-50:]}")
    
    # 检查是否与基准一致
    all_match_baseline = all(r == baseline for r in concurrent_results)
    
    if all_match_baseline:
        print("\n✅ PASS: 所有并发 greedy 结果与单请求基准一致")
    else:
        print("\n❌ FAIL: 并发 greedy 结果与单请求基准不一致！")
        print(f" 检测到 {len(set(concurrent_results))} 种不同的输出：")
        mismatched = [(len(r), r[-50:]) for r in concurrent_results if r != baseline]
        print(f"   不一致的结果: {mismatched}")
        
if __name__ == "__main__":
    # 运行测试
    try:
        test_single_vs_concurrent()
    except Exception as e:
        print(f"测试执行出错: {e}")
        print("请确保 vLLM 服务已启动在 http://localhost:8102")
```
## Token vs 字符长度 的差异

| 对比项 | vLLM `completion_tokens` | Python `len(response)` |
|--------|------------------------|----------------------|
| **统计单位** | **Token** (BPE/WordPiece) | **字符/字节** |
| **中文** | 1个汉字 ≈ 1-2 tokens | 1个汉字 = 1个字符 |
| **英文** | 1个单词 ≈ 1-2 tokens | 按字符数计算 |
| **数字/标点** | 可能合并为1个token | 每个单独计数 |

---

## 典型差异比例

| 语言/内容 | Token数 : 字符长度 |
|-----------|-------------------|
| 英文文本 | 1 : 4 ~ 1 : 6 |
| 中文文本 | 1 : 1 ~ 1 : 1.5 |
| 代码/数字 | 1 : 3 ~ 1 : 10 (数字很长时) |

---

## 实际影响计算

假设你的场景：
```python
# vLLM 返回
completion_tokens = 380

# Python 统计 (假设是中英文混合)
response_text = "..."  # len = 500 字符

# 你的错误计算 (如果用字符长度当token)
wrong_throughput = 500 / costtime  # 偏大!
correct_throughput = 380 / costtime  # 正确
```

**如果误用字符长度，吞吐量会被高估 1.3x ~ 6x！**

---

## 验证方法

在你的 Python 统计代码里加这个：

```python
import tiktoken  # OpenAI 的 tokenizer

# 获取 vLLM 实际用的 tokenizer (通常是 Llama/T5/GPT2 等)
# 方法1: 直接用 response.usage
completion_tokens = response.usage.completion_tokens  # 准确的token数

# 方法2: 本地计算 (如果模型是 Llama2/3)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your-model")

tokens = tokenizer.encode(response_text)
local_token_count = len(tokens)

print(f"vLLM统计: {completion_tokens}")
print(f"本地tokenize: {local_token_count}")
print(f"字符串长度: {len(response_text)}")
print(f"比例: {len(response_text)/completion_tokens:.2f}")
```

---

## 快速诊断你的问题

请检查你的统计代码：

```python
# ❌ 错误的 (会导致统计偏高)
token_count = len(response_text)  # 字符长度!  

# ✅ 正确的
token_count = response.usage.completion_tokens  # vLLM返回的token数
```

**你是不是用了 `len(response_text)` 来计算 token 数？** 如果是，这就是吞吐量偏高的原因！

## 修正后的对比

| 指标 | 数值 |
|------|------|
| 字符长度 | ~500-600 |
| 实际 tokens | 380 |
| **真实吞吐** | 380 / 27.59s = **13.77 t/s** |
| vLLM 日志估算 | 412.7 / 32 = **12.9 t/s** |

这样就**基本吻合**了！
