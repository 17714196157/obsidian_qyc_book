## 📋 问题总结

这是一个 **vLLM 0.4.3** 版本的 Bug：**并发请求时，greedy 解码（temperature=0）的结果会受到其他采样请求的影响**，导致结果不一致。
### 核心问题
- 当 `temperature=0`（贪婪解码）时，结果应该是**完全确定性的**
- 但并发执行时，greedy 请求的结果会被其他带有采样参数（如 `seed`、`top_k`）的请求"污染"
- 表现为同样的 greedy 请求返回了不同的结果（如 `old man` vs `young woman`）
## 📊 检测方法

| 现象                        | 结论         |
| ------------------------- | ---------- |
| 所有 greedy 请求返回**完全相同**的结果 | ✅ 无此 Bug   |
| greedy 请求返回**2种或以上**不同结果  | ❌ 存在此 Bug  |
| greedy 结果与单请求基准**不一致**    | ❌ 存在并发污染问题 |

---

## 🔍 如何检测你的 vLLM 是否有同样问题

### 方法一：快速测试脚本（推荐）

使用下面的测试脚本直接验证：

```python
import json
import concurrent.futures
import requests
import time
api_url = "http://192.168.0.172:8000/v1/completions"  # "http://192.168.0.172:8102/v1/completions"  # 修改为你的 vLLM 地址
prompt = "Once upon a time,"
max_tokens = 100
model = "/home/qyc/bert/Qwen2-0.5B"  # "qwq-32b"
  
def post_http_request(api_url: str, pload, request_id: int) -> dict:
    headers = {"User-Agent": "Test Client"}
    try:
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
        data = json.loads(response.content)
        # output = data.get("text", [""])[0] if isinstance(data.get("text"), list) else data.get("text", "")
        output = data.get("choices", [{'text': ''}])[0]["text"]
        # request_id = data.get("id", "000")
  
        return {"id": request_id, "output": output, "params": pload}
    except Exception as e:
        return {"id": request_id, "error": str(e)}
  
def test_concurrent_greedy_consistency():
    """
    检测并发请求时 greedy 解码是否一致
    """
    # api_url = "http://192.168.0.172:8102/v1/completions"  # 修改为你的 vLLM 地址
    # prompt = "Once upon a time,"
    # 构建请求参数
    sampling_params = []
    # 添加 greedy 请求 (temperature=0, 应该总是返回相同结果)
    for i in range(10):
        sampling_params.append({
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "model": model,
            "request_id": f"greedy_{i}"
        })
    # 添加带采样的请求 (可能干扰 greedy)
    for i in range(10):
        sampling_params.append({
            "prompt": prompt,
            "model": model,
            "seed": 42 + i,
            "max_tokens": 10,
            "top_p": 0.9,
            "temperature": 0.7,
            "request_id": f"sampling_{i}"
        })
    print("=" * 60)
    print("开始并发测试...")
    print(f"总共 {len(sampling_params)} 个并发请求")
    print("=" * 60)
    # 并发执行所有请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i, param in enumerate(sampling_params):
            futures.append(
                executor.submit(post_http_request, api_url=api_url, pload=param, request_id=i)
            )
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"请求异常: {exc}")
    # 分析结果
    print("\n" + "=" * 60)
    print("结果分析")
    print("=" * 60)
    # 提取所有 greedy 请求的结果
    greedy_results = [r for r in results if "greedy" in str(r.get("params", {}).get("request_id", ""))]
    sampling_results = [r for r in results if "sampling" in str(r.get("params", {}).get("request_id", ""))]
    # 检查 greedy 结果的一致性
    greedy_outputs = [r.get("output", "") for r in greedy_results if "error" not in r]
  
    if len(set(greedy_outputs)) == 1:
        print("✅ PASS: 所有 greedy 请求返回了相同的结果")
        print(f"   统一结果: {greedy_outputs[0][:50]}...")
    else:
        print("❌ FAIL: greedy 请求返回了不一致的结果！")
        print(f"   检测到 {len(set(greedy_outputs))} 种不同的输出:")
        for i, output in enumerate(set(greedy_outputs)):
            count = greedy_outputs.count(output)
            print(f"   结果 {i+1} (出现 {count} 次): {output[:60]}...")
    # 显示贪心greedy_outputs请求结果
    print("显示贪心greedy_outputs请求结果")
    print(f"\n贪心请求数量: {len(greedy_results)}")
    greedy_results = [r.get("output", "") for r in greedy_results if "error" not in r]
    print(f"贪心结果多样性: {len(set(greedy_results))} 种不同输出")
    # 显示采样请求结果（参考）
    print("显示采样sampling_results请求结果")
    print(f"\n采样请求数量: {len(sampling_results)}")
    sampling_outputs = [r.get("output", "") for r in sampling_results if "error" not in r]
    print(f"采样结果多样性: {len(set(sampling_outputs))} 种不同输出")
    return len(set(greedy_outputs)) == 1
  
def test_single_vs_concurrent():
    """
    对比单请求 greedy 和并发 greedy 的结果
    """
    # api_url = "http://192.168.0.172:8102/v1/completions"  # 修改为你的 vLLM 地址
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
        "max_tokens": max_tokens
    }
    response = requests.post(api_url, headers={"User-Agent": "Test"}, json=baseline_payload, stream=True)
    data = json.loads(response.content)
    # baseline = data.get("text", [""])[0] if isinstance(data.get("text"), list) else data.get("text", "")
    baseline = data.get("choices", [{'text': ''}])[0]["text"]
    print(f"   基准结果: {baseline}")
    # 然后并发发多个 greedy 请求
    print("\n2. 并发发送 5 个 greedy 请求...")
    time.sleep(1)  # 等待一下
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(5):
            futures.append(
                executor.submit(post_http_request, api_url=api_url, pload=baseline_payload, request_id=i)
            )
        concurrent_results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            concurrent_results.append(result.get("output", ""))
            print(f"   并发结果 {result['id']}: {result.get('output', '')}")
    # 检查是否与基准一致
    all_match_baseline = all(r == baseline for r in concurrent_results)
    if all_match_baseline:
        print("\n✅ PASS: 所有并发 greedy 结果与单请求基准一致")
    else:
        print("\n❌ FAIL: 并发 greedy 结果与单请求基准不一致！")
        mismatched = [r for r in concurrent_results if r != baseline]
        print(f"   不一致的结果: {mismatched}")
  
  
def test_single():
    """
    单请求 greedy
    """
    # api_url = "http://192.168.0.172:8102/v1/completions"  # 修改为你的 vLLM 地址
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
        "max_tokens": max_tokens
    }
    response = requests.post(api_url, headers={"User-Agent": "Test"}, json=baseline_payload, stream=True)
    data = json.loads(response.content)
    print(data)
    baseline = data.get("choices", [{'text': ''}])[0]["text"]
    print(f"   基准结果: {baseline}")
  
if __name__ == "__main__":
    # 运行测试
    try:
        # test_single()
        # test_concurrent_greedy_consistency()
        test_single_vs_concurrent()
    except Exception as e:
        print(f"测试执行出错: {e}")
        print("请确保 vLLM 服务已启动在 http://localhost:8102")
```

### 方法二：检查 vLLM 版本

```bash
# 查看当前 vLLM 版本
python -c "import vllm; print(vllm.__version__)"

# 或使用 pip
pip show vllm
```

**受影响版本**：根据 Issue，问题出现在 **vLLM 0.4.3** 及相近版本。

### 方法三：查看 Issue 修复状态

检查该 Issue 是否已关闭以及修复版本：

```bash
# 查看 vLLM 最新版本
pip index versions vllm
```

---

## 🛠️ 解决方案

如果检测发现你有同样的问题，可以尝试：

### 1. **升级 vLLM**
```bash
pip install --upgrade vllm
```

### 2. **临时规避方案**
- **避免混用**：不要将 greedy 请求（`temperature=0`）和采样请求（`temperature>0`）并发发送到同一实例
- **分离部署**：为 greedy 任务和采样任务分别部署独立的 vLLM 实例
- **串行处理**：对需要确定性的 greedy 请求进行串行处理

### 3. **使用 `--enforce-eager` 模式（可能有效）**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --enforce-eager  # 禁用 CUDA graph，可能避免状态污染
```

---
