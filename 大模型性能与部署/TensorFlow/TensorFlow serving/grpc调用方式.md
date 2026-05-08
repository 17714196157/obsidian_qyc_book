模型信息
在写Gprc服务之前，需要明确模型的名字、输入、输出等。我们使用curl http://localhost:8501/v1/models/half_plus_two/metadata可以看到Docker中模型的基本信息

```
http://120.27.159.54:8501/v1/models/half_plus_two/metadata
{
  "model_spec": {
    "name": "half_plus_two",
    "signature_name": "",
    "version": "1587612107"
  },
  "metadata": {
    "signature_def": {
      "signature_def": {
        "serving_default": {
          "inputs": {
            "Input-Segment:0": {
              "dtype": "DT_FLOAT",
              "tensor_shape": {
                "dim": [
                  {
                    "size": "-1",
                    "name": ""
                  },
                  {
                    "size": "64",
                    "name": ""
                  }
                ],
                "unknown_rank": false
              },
              "name": "Input-Segment:0"
            },
            "Input-Token:0": {
              "dtype": "DT_FLOAT",
              "tensor_shape": {
                "dim": [
                  {
                    "size": "-1",
                    "name": ""
                  },
                  {
                    "size": "64",
                    "name": ""
                  }
                ],
                "unknown_rank": false
              },
              "name": "Input-Token:0"
            }
          },
          "outputs": {
            "dense_1/Softmax:0": {
              "dtype": "DT_FLOAT",
              "tensor_shape": {
                "dim": [
                  {
                    "size": "-1",
                    "name": ""
                  },
                  {
                    "size": "2",
                    "name": ""
                  }
                ],
                "unknown_rank": false
              },
              "name": "dense_1/Softmax:0"
            }
          },
          "method_name": "tensorflow/serving/predict"
        }
      }
    }
  }
}

```

```
pip 下载grpc访问库 https://pypi.org/project/tensorflow-serving-api/1.14.0/#files
```

```python
#encoding=utf8
import requests
import numpy as np
import tensorflow.compat.v1 as tf
import time
tf.disable_v2_behavior()
# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(precision=3)

from keras.applications import xception
from tensorflow.python.platform import gfile

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

from kashgari import utils
from setting import saved_model_path
from pre_data import PreProcessData
processor = utils.load_processor(model_path=saved_model_path)

def prediction(sence, rsence_B_list):
    rsence_B_list = rsence_B_list[:3]
    X_train = [PreProcessData.dformat((sence, x)) for x in rsence_B_list]
    print(X_train)
    tensor = processor.process_x_dataset(X_train)
    tensor = [{
        "Input-Token:0": i.tolist(),
        "Input-Segment:0": np.zeros(i.shape).tolist()
    } for i in tensor]
    ids = np.array([x["Input-Token:0"] for x in tensor]).astype(np.float32)
    segment_ids = np.array([x["Input-Segment:0"] for x in tensor]).astype(np.float32)
    tensor =  np.array(tensor)
    # print(np.array(tensor))

    # predict 通过http接口
    t1 = time.time()
    if True:
        r = requests.post(
            "http://120.27.159.54:8501/v1/models/half_plus_two:predict",
            json={"instances": tensor.tolist()})
        preds = r.json()['predictions']

    result_list = []
    for index, (x, pred) in enumerate(zip(rsence_B_list, preds)):
        result_list.append((x, pred[1]))  # 取出第二个标签 对应的概率值
    t2 = time.time()
    print(f"1111111 {result_list} {str(t2-t1)}")

    # ------ Only for BERT Embedding End ----------
    # predict 通过grpc接口
    t1 = time.time()
    channel = grpc.insecure_channel('120.27.159.54:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'half_plus_two' #对应上图第一个方框
    request.model_spec.signature_name = 'serving_default' #对应上图第二个方框
    # print(f"ids.shape:{ids.shape}")  # ids:输入批量句子，每个句子的字符编码 ;
    request.inputs['Input-Token:0'].CopyFrom(
        tf.make_tensor_proto(ids, shape=[ids.shape[0], ids[0].size], dtype=tf.float32))  # shape: 数据每个维度的大小， 【行数、每行元素个数】
    request.inputs['Input-Segment:0'].CopyFrom(tf.make_tensor_proto(segment_ids, shape=[segment_ids.shape[0], segment_ids[0].size] , dtype=tf.float32)) #对应上图第三个方框，为模型的输入Name
    # print(request)

    result_future = stub.Predict.future(request)  # 10 secs timeout
    result = result_future.result()

    result_dict = {}
    for key in result.outputs:
        tensor_proto = result.outputs[key]
        nd_array = tf.make_ndarray(tensor_proto)
        result_dict[key] = nd_array
    preds = []
    for (x, pred) in zip(rsence_B_list, result_dict['dense_1/Softmax:0']):
        preds.append(pred[1])

    print(preds)
    t2 = time.time()
    print(f"222222 {result_list} {str(t2-t1)}")


if __name__ == "__main__":
    sence= "左侧单侧乳房根治性切除术"
    load_data_obj = PreProcessData(icdcode_filename="code-{}手术编码表.txt".format("临床版"))
    rsence_B_list=[x for x in load_data_obj.choise_sim_sample(sence, samplen1=5, samplen2=5, threshold=0.4)]
    prediction(sence, rsence_B_list)


```

代码示例2： 生成模型预测过程中，使用tfserving
模型部署展示：
![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/grpc调用方式/4cf7dcbf11651193cd67e68b1b2a42b1_MD5.png]]

![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/grpc调用方式/bb3cfafd257a19900ca9643488e41515_MD5.png]]
启动tfserving  docker run -dt -p 8501:8501 -p 8500:8500  -v "/home/qyc/multiModel:/models/multiModel" -e MODEL_NAME=multiModel tensorflow/serving --model_config_file=/models/multiModel/models.config

```
(base) ➜  multiModel cat models.config
model_config_list: {
    config: {
        name: "operation",
        base_path: "/models/multiModel/operation",
        model_platform: "tensorflow",
        }
}
```

```python
#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
from __future__ import print_function
import numpy as np
import pandas as pd
import tqdm
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
import traceback
import re
import os
import sys
import time
import requests

base_path = r'/home/qyc/generateSurg'
bert_model_dir = r'/home/qyc/generateSurg/bert'
sys.path.append(base_path)

from setting import bert_model_dir, model_type, save_model_dir, base_path, logger, input_path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# 基本参数
maxlen = 256
batch_size = 8
epochs = 60  # 10000
use_simply= True  # 简约版albert，自定义层


# bert配置
from bert4keras.backend import keras, set_gelu
set_gelu('tanh')  # 切换gelu版本

if use_simply is True:
    config_path = os.path.join(bert_model_dir, r'albert_config_c.json')
else:
    config_path = os.path.join(bert_model_dir, r'albert_config.json')


checkpoint_path = os.path.join(bert_model_dir, r'model.ckpt-best')
dict_path = os.path.join(input_path, r'vocab_chinese.txt')


# 加载并精简词表，建立分词器
token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    # startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
keep_tokens = None

# 补充词表
compound_tokens = []
new_token_dict = token_dict.copy()
additional_special_tokens = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ", "秕", "襞", "瘭", "髌", "蝽", "毳", "菪", "镫", "鍉", "耵", "哚", "蒽", "跗", "腘", "骺", "钬", "嵴", "肼", "皲", "颏", "髁", "疬", "蠊", "莨", "蛉", "癃", "瘰", "蝥", "𧿹", "踇", "蛲", "聍", "胬", "搦", "袢", "襻", "铍", "胼", "蜣", "鲭", "蝾", "朊", "铯", "螫", "铊", "酞", "羰", "缬", "氩", "癔", "铟", "吲", "蚰", "蚴", "纡", "螈", "谵" ]
i = 2 # 只占用 【unused97】的编号位置, 不能超过 21128
for token in additional_special_tokens:
    if token not in new_token_dict:
        compound_tokens.append([i])
        new_token_dict[token] = i
        i+=1

# 建立分词器
tokenizer = Tokenizer(new_token_dict, do_lower_case=True)


model = build_transformer_model(
    config_path,
    checkpoint_path,
    model="albert",     # model="bert",
    application='unilm',
    compound_tokens = compound_tokens  # 增加新token，用旧token平均来初始化
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (content, title) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


from bert4keras.layers import Loss
class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


from keras.models import Model
output = CrossEntropy(2)(model.inputs + model.outputs)
model = Model(model.inputs, output)



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

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        logger.debug("use predict")
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        t1 = time.time()
        res = self.last_token(self.model).predict([token_ids, segment_ids])
        # res = self.model.predict([token_ids, segment_ids])[:,-1]

        t2 = time.time()
        logger.debug(f"predict func model.predict cost time={t2-t1}")
        return res

    def generate(self, text, topk=1, flag_ort=False):
        t1 = time.time()
        max_c_len = maxlen - self.maxlen
        # print(f"text={text} max_c_len={max_c_len}")
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)

        predict_fun = self.predict

        if flag_ort is False:
            predict_fun = self.predict  # 基于beam search
        else:
            self.requests_sess = requests.Session() #  requests 设置为 长链接

            predict_fun = self.predict_ort # 基于beam search

        output_ids = self.beam_search([token_ids, segment_ids], topk=topk, predict_fun=predict_fun)  # 基于beam search
        t2 = time.time()
        logger.debug(f"generate flag_ort:{flag_ort} cost time: {t2-t1}")
        return tokenizer.decode(output_ids)

    def beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1, predict_fun=None):
        """beam search解码
        说明：这里的topk即beam size；
        返回：最优解码序列。
        """
        if predict_fun is None:
            raise IOError(f"predict_fun is not define")

        inputs = [np.array([i]) for i in inputs]
        output_ids, output_scores = self.first_output_ids, np.zeros(1)
        for step in range(self.maxlen):

            scores, states = predict_fun(
                inputs, output_ids, states, temperature, 'logits'
            )  # 计算当前得分

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
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best = output_scores.argmax()  # 得分最大的那个
                if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
                    return output_ids[best]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]


    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict_ort(self, inputs, output_ids, states):
        # logger.debug("use predict_ort")
        # token_ids, segment_ids = inputs
        # token_ids = np.concatenate([token_ids, output_ids], 1)
        # segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        # onnx_input = {"Input-Token:0": to_numpy(token_ids).astype(np.float32),
        #               "Input-Segment:0": to_numpy(segment_ids).astype(np.float32),
        #               }
        # t1 = time.time()
        # # res = self.Onnx_sess.run(['cross_entropy_1/Identity:0'], onnx_input)[0]
        # res = self.Onnx_sess.run(None, onnx_input)[0]
        # t2 = time.time()
        # logger.debug(f"predict_ort MODEL.run cost time = {t2-t1}")
        # res = res[:, -1]

        logger.debug("use predict_tfserving_http")
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)

        onnx_input = {"Input-Token:0": token_ids[0].tolist(),
                      "Input-Segment:0": segment_ids[0].tolist(),
                      }
        # print(onnx_input)
        # predict 通过http接口 tfserving 方式部署 生成模型, 那么每次返回[1,当前序列有效长度, vocab_size]
        t1 = time.time()
        r = self.requests_sess.post(
            "http://192.168.0.5:8501/v1/models/operation:predict",
            json={"instances": [onnx_input]})
        # print( r.json())
        preds = r.json()['predictions']
        preds = np.array(preds)
        res = preds[:, -1, :]
        # print(preds.shape, res.shape, np.argmax(res,axis=1), tokenizer.decode(np.argmax(res,axis=1)) )
        # (1, 32, 21197) (1, 21197) [1381] 右

        t2= time.time()
        logger.debug(f"predict func model.predict cost time={t2-t1}")

        return res


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_bleu = 0.
        self.lowest = 1e10
        self.lowest_trainloss = 1e10


    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        # if logs['loss'] <= self.lowest:
        #     self.lowest = logs['loss']
        #     model.save_weights(model_path)
        #
        if logs['val_loss'] <= self.lowest:
            self.lowest = logs['val_loss']
            self.lowest_trainloss = logs['loss']
            model.save_weights(model_path)


        logger.debug(f"loss={logs['loss']} val_loss={logs['val_loss']}  self.lowest ={self.lowest}  self.lowest_trainloss ={self.lowest_trainloss}")

        # 演示效果
        self.just_show()


    def just_show(self):
        if model_type == "new_operation":
            s1 = u'左侧经皮肾镜钬激光碎石术+左侧输尿管支架置入术&左侧经皮肾镜钬激光碎石术'  ## 腹腔镜下阑尾切除术

        elif model_type == "tumour":
            s1 = u'胃粘液腺癌介入化疗术后伴多发转移'  ## 胃粘液腺癌&粘液腺癌多发转移

        elif model_type == "injury":
            s1 = u'左膝关节内、外侧半月板后角撕裂'  ## 胃粘液腺癌&粘液腺癌多发转移

        elif model_type == "inaccuracy_diagnosis":
            s1 = u'上消化道出血消化性溃疡伴出血？消化道肿瘤？'  ## 胃粘液腺癌&粘液腺癌多发转移

        for s in [s1,]:
            print(s, f' 生成标题:', autotitle.generate(s))




if __name__ == '__main__':
    from tool.memorymodels import get_model_sample_histroy_dict, find_like_sample
    txts = get_model_sample_histroy_dict(model_type, return_data_type="list")
    pattern = "|".join(additional_special_tokens)
    from sklearn.model_selection import train_test_split
    txts_train, txts_eval = train_test_split(txts, test_size=0.02, random_state=7, shuffle=True)

    # txts_train, txts_eval = txts, txts
    logger.debug(txts_train[:3], txts_eval[:3])
    logger.debug(f"txts_train={len(txts_train)} txts_eval={len(txts_eval)} ")
    logger.debug(f"tokenizer._token_end_id={tokenizer._token_end_id}")
    autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)

    # 加入 优化器， 编译模型文件
    model.compile(optimizer=Adam(2e-5))
    model.summary()

    try:
        model_path = os.path.join(save_model_dir, 'simply_seq2seq_surg_{}.weights'.format(model_type))
        logger.info(f"model.load_weights model_path:{model_path}")
        model.load_weights(model_path)
        autotitle.model = model



    except Exception as e:
        logger.error(f"加载之前模型训练结果失败 {e}")
        model_path = os.path.join(save_model_dir, 'simply_seq2seq_surg_{}.weights'.format(model_type))

        autotitle.model = model
        pass


    evaluator = Evaluator()
    train_generator = data_generator(txts_train, batch_size)
    eval_generator = data_generator(txts_eval, batch_size)


    # 基本参数
    steps_per_epoch_train = max(int(float(len(txts_train) / batch_size)), 1)
    steps_per_epoch_eval = max(int(float(len(txts_eval) / batch_size)), 1)
    logger.info(f"steps_per_epoch={steps_per_epoch_train} epochs={epochs} batch_size={batch_size}")

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", update_freq=1)

    t1 = time.time()
    model.fit_generator(train_generator.forfit(),
                        validation_data=eval_generator.forfit(),validation_steps=len(eval_generator),
                        steps_per_epoch=steps_per_epoch_train, # steps_per_epoch,
                        epochs=epochs,
                        callbacks=[evaluator,])
    model.save_weights(model_path+"_end")

    t2 = time.time()
    print(f"训练总耗时 cost time ={t2-t1}")
    # # 保存成save-model
    # utils.convert_to_saved_model(model=model, model_path=os.path.join(save_model_dir, '{}'.format(model_type)))

else:
    logger.debug(f"tokenizer._token_end_id={tokenizer._token_end_id}")
    autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)

    model_path = os.path.join(save_model_dir, 'simply_seq2seq_surg_{}.weights'.format(model_type))
    model.load_weights(model_path)
    autotitle.model = model
    print(model.summary())


    def to_numpy(tensor):
        try:
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            return tensor.cpu().numpy()
    
        except Exception as e:
            return tensor


    text = "右手中指清创+神经血管肌腱探查修复术&神经血管肌腱探查修复术"
    # beam_search() 逐步生成过程 预测的代码过程：  直接模型预测
    t1 = time.time()
    output = autotitle.generate(text, flag_ort=False)
    t2 = time.time()
    print(output)
    print(f"bert4keras cost time={t2-t1}")
    print("tf.test.is_gpu_available():", tf.test.is_gpu_available())


    # # beam_search() 逐步生成过程 预测的代码过程：  tfserving部署预测
    t1 = time.time()
    output = autotitle.generate(text, flag_ort=True)
    t2 = time.time()
    print(output)
    print(f"onnx_predict cost time={t2-t1}")











```