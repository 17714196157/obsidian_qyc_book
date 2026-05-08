TensorFlow Servering在线部署模型
     serving镜像提供了两种调用方式：gRPC和HTTP请求。gRPC默认端口是8500，HTTP请求的默认端口是8501，
serving镜像中的程序会自动加载镜像内/models下的模型，通过MODEL_NAME指定/models下的哪个模型

下载容器   docker pull tensorflow/serving
下载GPU容器  docker pull tensorflow/serving:latest-gpu

下载demo例子：
```
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone https://github.com/tensorflow/serving

```



启动自己的模型:
```
docker run -dt -p 8501:8501  -v "/home/qyc/workcode/saved_model:/models/half_plus_two" -e MODEL_NAME=half_plus_two tensorflow/serving
```
![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/TensorFlow serving在线部署模型/c96ae760b3a54e2ddb986c04b3e3c23c_MD5.png]]
```
curl -d '{"instances": [101,6117,5052,1217,1327,1177,4118,3800,102]}' -X POST http://localhost:8521/v1/models/half_plus_two:predict
或者
docker run -p 8501:8501 --mount type=bind,source=/home/qyc/workcode/saved_model,target=/models/half_plus_two -e MODEL_NAME=half_plus_two -t tensorflow/serving  &
```

检查端口服务:
注意： http接口默认只能8501，除非docker启动时带上配置文件
curl http://localhost:8501/v1/models/half_plus_two
![[d285687d8b242f05206975cc61198613_MD5.png]]

```python
代码调用示例：
import requests
from kashgari import utils
import numpy as np
from Icd import OptIcd

sence = "血管加压剂灌注"
x = OptIcd.get_word_list(sence)
print("x:", x)
# Pre-processor data
processor = utils.load_processor(model_path=r'D:\code\手术程序\icd2_classify\saved_model_icd2\1583888484')
tensor = processor.process_x_dataset([x])
print("tensor:", tensor)
# array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)

# if you using BERT, you need to reformat tensor first
# ------ Only for BERT Embedding Start --------
tensor = [{
   "Input-Token:0": i.tolist(),
   "Input-Segment:0": np.zeros(i.shape).tolist()
} for i in tensor]
# ------ Only for BERT Embedding End ----------
tensor = np.array(tensor)
print("tensor:", tensor)

# predict
# docker run -it -d -p 8521:8521 -v source:/home/qyc/workcode/程序上传/saved_model -v target:/models/half_plus_two -e MODEL_NAME=half_plus_two tensorflow/serving
r = requests.post("http://localhost:8521/v1/models/half_plus_two:predict", json={"instances": tensor.tolist()})
preds = r.json()['predictions']
print("preds:\n{}".format(preds))
# Convert result back to labels
labels = processor.reverse_numerize_label_sequences(np.array(preds).argmax(-1))
print("labels:\n{}".format(labels))
# labels = ['video']
```
![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/TensorFlow serving在线部署模型/e48bc6cc58b59a8244667af9c3b699e7_MD5.png]]

demo尝试测试：
1. 挂载模型文件，启动docker
```
docker run -dt -p 8501:8501 -v "/home/qyc/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu:/models/half_plus_two" -e MODEL_NAME=half_plus_two tensorflow/serving

```
1. 切入docker 可以看到挂的模型文件
```
docker exec -it kind_ardinghelli bash
```

![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/TensorFlow serving在线部署模型/f9c3812113d2ab9d582c826e7a4743bf_MD5.png]]
3. 请求模型预测
```
 curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/TensorFlow serving在线部署模型/39fed7d4485c9bd5189a4f024e1b8ec5_MD5.png]]

![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/TensorFlow serving在线部署模型/730e49bf55b633d1ad4dae2d0bde8606_MD5.png]]

![[大模型性能与部署/TensorFlow/TensorFlow serving/assets/TensorFlow serving在线部署模型/8ab2c03ffc5e43e6082e2ddb689860ba_MD5.png]]



docker exec -it  sweet_robinson bash   # 可以进容器看一下 能再对应目录找到文件 就没挂失败   

查看docker内存使用
docker stats --no-stream --format "{\"container\":\"{{ .Container }}\",\"memory\":{\"raw\":\"{{ .MemUsage }}\",\"percent\":\"{{ .MemPerc }}\"},\"cpu\":\"{{ .CPUPerc }}\"}"
或者：
![[52d7f19d69ee964a8768679bb398a1e9_MD5.png]]

