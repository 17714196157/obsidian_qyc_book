---
tags:
  - onnx
  - tensorflow
---

# TensorFlow 转 ONNX 部署

## 安装工具

```bash
pip install -U tf2onnx     # 用于将 Save_model 格式的 TensorFlow 模型文件转成 ONNX
pip install onnxruntime==1.4.0 或者 pip install onnxruntime-gpu==1.4.0   # 用于部署 ONNX
```

> ⚠️ **特别注意**：onnxruntime 的版本号与 CUDA 的版本有强依赖关系。

---

## 转换步骤

### 1. 将 Keras/TF 模型保存成 SavedModel 格式
![[5adcf7f885c1db2e241c42521d442259_MD5.png]]
```python
from tensorflow.python import keras, saved_model
def convert_to_saved_model(model,
                           model_path: str,
                           version: str = None,
                           inputs: Optional[Dict] = None,
                           outputs: Optional[Dict] = None):
    """
    Export model for tensorflow serving
    Args:
        model: Target model
        model_path: The path to which the SavedModel will be stored.
        version: The model version code, default timestamp
        inputs: dict mapping string input names to tensors. These are added
            to the SignatureDef as the inputs.
        outputs:  dict mapping string output names to tensors. These are added
            to the SignatureDef as the outputs.
    """
    pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
    if version is None:
        version = round(time.time())
    export_path = os.path.join(model_path, str(version))

    if inputs is None:
        inputs = {i.name: i for i in model.inputs}
    if outputs is None:
        outputs = {o.name: o for o in model.outputs}

    print(f"model={model}")
    # >>><keras.engine.training.Model object at 0x0000016EAA628400>
    print(f"inputs={inputs}")
    # >>>inputs={'Input-Token:0': <tf.Tensor 'Input-Token:0' shape=(?, ?) dtype=float32>, 'Input-Segment:0': <tf.Tensor 'Input-Segment:0' shape=(?, ?) dtype=float32>}
    print(f"outputs={outputs}")
    # >>>outputs={'MLM-Proba/truediv:0': <tf.Tensor 'MLM-Proba/truediv:0' shape=(?, ?, 21128) dtype=float32>}
    print(f"export_path={export_path}") # export_path=D:\code\workcode\generateSurg\save\new_operation\1641306010
    sess = keras.backend.get_session()
    saved_model.simple_save(session=sess,
                            export_dir=export_path,
                            inputs=inputs,
                            outputs=outputs)
    return saved_model
```

### 2. SavedModel 格式转成 ONNX
```bash
python -m tf2onnx.convert --saved-model  /home/qyc/generateSurg/save/new_operation/1641868208   --output  etnet.onnx --opset 11
```

**转换输出：**
```
2022-01-11 10:33:35,879 - INFO - Successfully converted TensorFlow model /home/qyc/generateSurg/save/new_operation/1641868208 to ONNX
2022-01-11 10:33:35,879 - INFO - Model inputs: ['Input-Token:0', 'Input-Segment:0']
2022-01-11 10:33:35,879 - INFO - Model outputs: ['MLM-Proba/truediv:0']
2022-01-11 10:33:35,879 - INFO - ONNX model is saved at etnet.onnx
```

### 3. 使用 Netron 查看 ONNX 模型
[[9949c3118ab2b24680612b34441eb00d_MD5.png|Open: file-20260508221037400.png]]
![[9949c3118ab2b24680612b34441eb00d_MD5.png]]
```python
netron.start("/home/qyc/generateSurg/utils/etnet.onnx" , address=('192.168.0.5',8888))
```

打开 URL：`http://192.168.0.5:8888/`

> 💡 通过 Netron 可以查看 ONNX 文件模型，注意到版本输入的数据类型是 float32 和输入字段的名字，构建 onnx_input 时要保持一致。

### 4. 加载 ONNX 模型进行预测

> 代码介绍：加载训练好的 bert4keras 构建的生成模型，比较它的预测耗时与 ONNX 用 onnxruntime 推理耗时的情况。

```python
import pandas as pd
import os
import re
import platform
import copy
import pathlib
import json
from typing import List, Optional, Dict, Union
import time
import numpy as np

import sys
base_path = r'/home/qyc/generateSurg'
sys.path.append(base_path)
bert_model_dir = os.path.join(base_path, "bert", "albert_large")
save_model_dir = os.path.join(base_path, "save")
model_type = "new_operation"
from setting import base_path, bert_model_dir, save_model_dir

from bert4keras.backend import keras, set_gelu
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
import torch

# 限制内存的使用量
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


# 使用bert4keras定义生成模型的网络结构
set_gelu('tanh')  # 切换gelu版本
config_path = os.path.join(bert_model_dir, r'albert_config.json')
checkpoint_path = os.path.join(bert_model_dir, r'model.ckpt-best')
dict_path = os.path.join(bert_model_dir, r'vocab_chinese.txt')

token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    # startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)
maxlen = 128

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model="albert",
    application='unilm',
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)
# model.summary()
# 交叉熵作为loss，并mask掉输入部分的预测
y_true = model.input[0][:, 1:]  # 目标tokens
y_mask = model.input[1][:, 1:]
y_pred = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
model.add_loss(cross_entropy)

# 加载keras的模型
model_path = os.path.join(save_model_dir, 'qqtry_best_model_seq2seq_surg_{}.weights'.format(model_type))
print(f"model.load_weights model_path:{model_path}")
model.load_weights(model_path)
print(model.inputs)
print(model.outputs)


# 加载onnx的模型文件
import onnxruntime as ort
onnx_model_file_path = os.path.join(base_path, "utils", "etnet.onnx")
MODEL = ort.InferenceSession(onnx_model_file_path) # , providers=["CUDAExecutionProvider"]

def to_numpy(tensor):
    try:
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.cpu().numpy()

    except Exception as e:
        return tensor

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    # @AutoRegressiveDecoder.wraps('probas')
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)


        res = model.predict([token_ids, segment_ids])

        res = res[:, -1]
        return res

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict_ort(self, inputs, output_ids, step):

        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)

        # res = model.predict([token_ids, segment_ids])
        # t1 = time.time()
        onnx_input = {"Input-Token:0": to_numpy(token_ids).astype(np.float32),
                      "Input-Segment:0": to_numpy(segment_ids).astype(np.float32),
                      }
        res = MODEL.run(None, onnx_input)[0]
        # t2 = time.time()
        # print(f"predict_ort cost time = {t2-t1}")

        res = res[:, -1]
        return res

    def generate(self, text, topk=3, tokenizer=None, flag_ort=False):
        max_c_len = maxlen - self.maxlen
        # text = replace_to_trans_vocab(text, flag="换装")

        token_ids, segment_ids = tokenizer.encode(text, max_length=max_c_len)
        #print(token_ids)
        #print(segment_ids)

        output_ids = self.beam_search([token_ids, segment_ids], topk, flag_ort=flag_ort)  # 基于beam search
        # print(f"generate cost time = {t2 - t1}")
        # print(output_ids)

        output_str = tokenizer.decode(output_ids)
        output_str = output_str.replace("\t","").replace(" ","")

        # output_str = replace_to_trans_vocab(output_str, flag="换回")

        return output_str

    def beam_search(self, inputs, topk, flag_ort=False):
        """beam search解码
        说明：这里的topk即beam size；
        返回：最优解码序列。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids, output_scores = self.first_output_ids, np.zeros(1)
        for step in range(self.maxlen):

            # 选择是否使用 onnx模型加上推理
            if flag_ort is False:
                scores = self.predict(inputs, output_ids, step, 'logits')  # 计算当前得分
            else:
                scores = self.predict_ort(inputs, output_ids, step, 'logits')


            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]

            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
            indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
            indices_1 = indices // scores.shape[1]  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = np.concatenate([output_ids[indices_1], indices_2],
                                        1)  # 更新输出
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  # 更新得分
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best_one = output_scores.argmax()  # 得分最大的那个
                if indices_2[best_one, 0] == self.end_id:  # 如果已经终止
                    return output_ids[best_one]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = (indices_2[:, 0] != self.end_id)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]


autotitle = AutoTitle(start_id=None,
                      end_id=tokenizer._token_end_id,
                      maxlen=64)



def test_bert4keras_predict(text = "周围神经缝合术&周围神经缝合术"):
    # # 单次一步预测的代码过程：
    # token_ids, segment_ids = tokenizer.encode(text, max_length=64)
    # print(f"token_ids={token_ids}")
    #
    # token_ids = torch.tensor(token_ids).unsqueeze(0)
    # segment_ids = torch.tensor(segment_ids).unsqueeze(0)
    # t1 = time.time()
    # output = model.predict([token_ids, segment_ids])
    # t2 = time.time()
    # print(output.shape)
    # print(f"cost time={t2-t1}")

    # # beam_search() 逐步生成过程 预测的代码过程：
    t1 = time.time()
    output = autotitle.generate(text, tokenizer=tokenizer,flag_ort=False)
    t2 = time.time()
    print(output)
    print(f"bert4keras cost time={t2-t1}")

    print("tf.test.is_gpu_available():", tf.test.is_gpu_available())

def test_onnx_predict(text = "周围神经缝合术&周围神经缝合术"):
    # # 单次一步预测的代码过程：
    # # data_node = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # # input_ids = torch.tensor(data_node["input_ids"]).unsqueeze(0)
    # # attention_mask = torch.tensor(data_node["attention_mask"]).unsqueeze(0)
    #
    # token_ids, segment_ids = tokenizer.encode(text, max_length=64)
    # print(f"token_ids={token_ids}")
    # token_ids = torch.tensor(token_ids).unsqueeze(0)
    # segment_ids = torch.tensor(segment_ids).unsqueeze(0)
    #
    # t1 = time.time()
    # onnx_input = {"Input-Token:0": to_numpy(token_ids).astype(np.float32),
    #               "Input-Segment:0": to_numpy(segment_ids).astype(np.float32),
    #               }
    # output = MODEL.run(None, onnx_input)
    # t2 = time.time()
    # print(f"cost time = {t2-t1}")
    # print(f"onnx_input={onnx_input}")
    # print(output.shape)

    # # beam_search() 逐步生成过程 预测的代码过程：
    t1 = time.time()
    output = autotitle.generate(text, tokenizer=tokenizer,flag_ort=True)
    t2 = time.time()
    print(output)
    print(f"onnx_predict cost time={t2-t1}")


if __name__ == '__main__':
    pass
    test_bert4keras_predict()
    test_onnx_predict()
```
