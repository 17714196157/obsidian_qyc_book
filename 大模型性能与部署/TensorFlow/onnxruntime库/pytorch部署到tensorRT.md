---
tags:
  - onnx
  - tensorrt
---

# PyTorch 部署到 TensorRT

> 以 PyTorch 情感分析模型转 ONNX，使用 ONNX Runtime 预测加速为例。

## 完整代码

```python
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
from torch.cuda import amp
import time


class BERTBaseUncase(torch.nn.Module):
    def __init__(self):
        super(BERTBaseUncase, self).__init__()

        pretrained = r"D:\code\workcode\externalCauses\bert\albert_chinese_small"  # 'voidful/albert_chinese_small'  # 使用small版本Albert
        config = AlbertConfig.from_pretrained(
            pretrained,
        )

        self.bert = AlbertModel.from_pretrained(
            pretrained,
            config=config,
        )

        self.tokenizer = BertTokenizer.from_pretrained(pretrained)

        self.bert_drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(config.hidden_size, 2)

        torch.nn.init.normal_(self.out.weight, std=0.01)

    @amp.autocast()
    def forward(self, input_ids, attention_mask, token_type_ids):
        _, o2 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        # print(f"o2 ={o2.shape}") # o2 =torch.Size([1, 384])
        bo = self.bert_drop(o2)
        output = self.out(bo)
        print(f"output={output.shape} {output}")
        return output


def test_model(model):
    """测试 PyTorch 模型推理"""
    model.eval()
    text = "周围神经缝合术"
    data_node = model.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    t1 = time.time()
    output = model(**data_node)
    t2 = time.time()
    print(f"cost time = {t2-t1} output={output}")


def convert_to_onnx(model):
    """将模型转换成 ONNX 格式

    ⚠️ 注意：opset_version 参数
    - TensorRT 8.X 版本：设置为 13
    - TensorRT 7.X 版本：设置为 12
    """
    model.eval()

    text = "周围神经缝合术"
    data_node = model.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64
    )
    input_ids = torch.tensor(data_node["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(data_node["attention_mask"]).unsqueeze(0)
    token_type_ids = torch.tensor(data_node["token_type_ids"]).unsqueeze(0)

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        "model.onnx",
        opset_version=12,  # 版本建议11以上比较稳定
        input_names=['ids', 'mask', 'token_type_ids'],
        output_names=['output'],
        dynamic_axes={
            'ids': {0: "batch_size"},
            'mask': {0: "batch_size"},
            'token_type_ids': {0: "batch_size"},
            'output': {0: "batch_size"},
        },
        enable_onnx_checker=False,
    )


def test_predict():
    """使用 ONNX Runtime 预测"""
    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.cpu().numpy()

    import onnxruntime as ort
    MODEL = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

    text = "周围神经缝合术"
    pretrained = r"D:\code\workcode\externalCauses\bert\albert_chinese_small"  # 使用small版本Albert
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    data_node = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64
    )
    input_ids = torch.tensor(data_node["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(data_node["attention_mask"]).unsqueeze(0)
    token_type_ids = torch.tensor(data_node["token_type_ids"]).unsqueeze(0)

    t1 = time.time()
    onnx_input = {
        "ids": to_numpy(input_ids),
        "mask": to_numpy(attention_mask),
        "token_type_ids": to_numpy(token_type_ids),
    }
    output = MODEL.run(None, onnx_input)
    print(output)
    t2 = time.time()
    print(f"cost time = {t2-t1}")


if __name__ == "__main__":
    model = BERTBaseUncase()       # 1. 创建模型
    test_model(model)              # 2. 测试模型，查看例子的输出
    convert_to_onnx(model)         # 3. 将模型转换成 ONNX
    test_predict()                 # 4. 使用 ONNX 预测，查看输出是否正确

    """
    输出结果对比：

    PyTorch 推理：
    output=torch.Size([1, 2]) tensor([[-0.2709,  0.0630]], grad_fn=<AddmmBackward>)
    cost time = 0.5821239948272705

    ONNX Runtime 推理：
    output=torch.Size([1, 2]) tensor([[-0.2709,  0.0630]], grad_fn=<AddmmBackward>)
    [array([[-0.27085018,  0.06297855]], dtype=float32)]
    cost time = 0.0029914379119873047

    加速效果：PyTorch 0.58s → ONNX Runtime 0.003s（约 194 倍加速）
    """
```

## 关键步骤说明

| 步骤 | 函数 | 说明 |
|------|------|------|
| 1 | `test_model()` | 测试 PyTorch 原始模型推理 |
| 2 | `convert_to_onnx()` | 导出 ONNX 格式模型 |
| 3 | `test_predict()` | 使用 ONNX Runtime 验证推理结果 |

## 注意事项

1. **opset_version 版本选择**：
   - TensorRT 8.X → `opset_version=13`
   - TensorRT 7.X → `opset_version=12`
   - 一般建议 ≥ 11 以保证稳定性
1. **dynamic_axes**：设置动态 batch_size，支持不同批量推理
2. **providers**：使用 `"CUDAExecutionProvider"` 启用 GPU 加速


![[大模型性能与部署/TensorFlow/onnxruntime库/assets/pytorch部署到tensorRT/d81ece6a6a0872fc81b4e035ab16e4df_MD5.png]]