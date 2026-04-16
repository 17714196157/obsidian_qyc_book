
### 适配vllm， 如果错误也能提取的usege信息
```python
from openai import OpenAI
import numpy as np
import os
import re
import json
# import pandas as pd
from icecream import ic
# https://bailian.console.aliyun.com/?tab=model#/model-market
from tqdm import tqdm
# 设置你的 API 密钥
api_key =  "sk-b5e02d8f907b42f98044391e97f854ab"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen2.5-14b-instruct"
# api_key =  "empty"
# base_url = "http://192.168.0.181:8000/v1"
# model = "/home/qyc/bert/Qwen2-0.5B"
client = OpenAI(api_key=api_key, base_url=base_url )
import tiktoken
from transformers import AutoTokenizer
"""
| 阶段     | 行为                                               | 是否有 usage    |
| ------ | ------------------------------------------------ | ------------ |
| 请求验证阶段 | 检查 `prompt_tokens + max_tokens <= max_model_len` | ❌ 无 usage    |
| 流式生成阶段 | 正常输出 token                                       | ✅ 最后返回 usage |
"""
class TokenCounter:
    def __init__(self, model_path_or_name: str, use_hf: bool = False):
        self.use_hf = use_hf
        if use_hf:
            # 使用 HuggingFace Tokenizer（推荐用于本地模型）
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_name,
                local_files_only=True,
                trust_remote_code=True
            )
        else:
            # 使用 tiktoken
            try:
                self.encoding = tiktoken.encoding_for_model(model_path_or_name)
            except KeyError:
                # 默认使用 cl100k_base
                self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        if self.use_hf:
            return len(self.tokenizer.encode(text))
        else:
            return len(self.encoding.encode(text))
        
# counter = TokenCounter(model_path_or_name=model, use_hf=True)
# ic(counter.count("你好，南京有那些景点"))
import re
def chat_with_usage_on_error(messages, max_tokens=512, **kwargs):
    """
    尝试请求，如果输入超长，从错误中解析 token 信息
    """
    result = {
        "success": False,
        "content": None,
        "usage": None,
        "error_info": None
    }
    
    try:
        # 尝试流式请求
        stream = client.chat.completions.create(
            model=kwargs.get("model", "your-model"),
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **{k: v for k, v in kwargs.items() if k != "model"}
        )
        
        # 正常处理流式响应
        content_parts = []
        final_usage = None
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
            
            if chunk.usage:
                final_usage = chunk.usage
        
        result.update({
            "success": True,
            "content": "".join(content_parts),
            "usage": final_usage
        })
        
    except Exception as e:
        error_msg = str(e)
        result["error_info"] = error_msg
        
        # 解析 vLLM 超长错误中的 token 信息
        # 错误格式1: "you requested 32161 tokens (23969 in the messages, 8192 in the completion)"
        # 错误格式2: "max_tokens must be at least 1, got -186."
        # 错误格式3: "This model's maximum context length is 32000 tokens. However, you requested..."
        
        error_data = {
            "is_context_length_error": False,
            "max_context_length": None,
            "requested_total_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "deficit": None
        }
        
        # 匹配 vLLM 标准错误格式
        # 格式: "you requested X tokens (Y in the messages, Z in the completion)"
        pattern1 = r'you requested (\d+) tokens \((\d+) in the messages, (\d+) in the completion\)'
        match = re.search(pattern1, error_msg)
        if match:
            error_data["requested_total_tokens"] = int(match.group(1))
            error_data["prompt_tokens"] = int(match.group(2))
            error_data["completion_tokens"] = int(match.group(3))
            error_data["is_context_length_error"] = True
        
        # 匹配最大上下文长度
        pattern2 = r"maximum context length is (\d+) tokens"
        match = re.search(pattern2, error_msg, re.IGNORECASE)
        if match:
            error_data["max_context_length"] = int(match.group(1))
            error_data["is_context_length_error"] = True
        
        # 匹配 "got -186" 格式（计算 deficit）
        pattern3 = r'got\s+(-?\d+)'
        match = re.search(pattern3, error_msg)
        if match:
            deficit = int(match.group(1))
            if deficit < 0:
                error_data["deficit"] = abs(deficit)
                error_data["is_context_length_error"] = True
        
        # 构造模拟的 usage（基于错误信息推算）
        if error_data["is_context_length_error"]:
            # 如果能提取到 prompt_tokens，使用它；否则标记为 unknown
            prompt_tokens = error_data["prompt_tokens"] or "unknown"
            
            simulated_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,  # 没有生成任何内容
                "total_tokens": prompt_tokens if isinstance(prompt_tokens, int) else "unknown",
                "max_context_length": error_data["max_context_length"],
                "requested_completion_tokens": max_tokens,
                "requested_total_tokens": error_data["requested_total_tokens"],
                "deficit_tokens": error_data["deficit"],
                "note": "Request rejected - context length exceeded"
            }
            result["usage"] = simulated_usage
        
        result["error_parse"] = error_data
    #####
    if not result["success"]:
        print(f"❌ 请求失败")
        print(f"错误信息: {result['error_info']}")
        print(f"\n📊 解析的 Token 信息:")
        print(f"  - 是否超长错误: {result['error_parse']['is_context_length_error']}")
        print(f"  - 最大上下文长度: {result['error_parse']['max_context_length']}")
        print(f"  - 输入 Prompt Tokens: {result['error_parse']['prompt_tokens']}")
        print(f"  - 请求的 Completion Tokens: {result['error_parse']['completion_tokens']}")
        print(f"  - 请求的总 Tokens: {result['error_parse']['requested_total_tokens']}")
        print(f"\n📝 模拟的 Usage:")
        print(f"  {result['usage']}")
    else:
        print(f"✅ 请求成功")
        print(f"Usage: {result['usage']}")
        
        
    return result
# ============ 使用示例 ============
# 1. 正常请求
result = chat_with_usage_on_error(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    model=model
)
# 2. 超长请求 - 可以获取到错误中的 token 信息
long_text = "这是一个超长的文本 " * 5000000  # 制造超长输入
result = chat_with_usage_on_error(
    messages=[{"role": "user", "content": long_text}],
    max_tokens=4096,
    model=model
)

```