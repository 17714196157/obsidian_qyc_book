
```python
import os
import pathlib
import time
import sys
#print('预测结果', autotitle.generate(u'社区获得性肺炎'))
base_path = r'/home/qyc/generateSurg'
sys.path.append(base_path)


from setting import (logger, save_model_dir)
from main.model_albert import autotitle, model
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
set_session(sess)
model_type = "new_operation"
# model_path = os.path.join(save_model_dir, 'best_model_seq2seq_surg_{}.weights'.format(cli_args.get("model_type")))
model_path = os.path.join(save_model_dir, 'simply_seq2seq_surg_{}.weights'.format(model_type))
logger.info(f"model_path:{model_path}")
model.load_weights(model_path)
autotitle.model = model



def convert_to_saved_model(model, model_path):
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
    version = round(time.time())

    export_path = os.path.join(model_path, str(version))

    inputs = {i.name: i for i in model.inputs}
    outputs = {o.name: o for o in model.outputs}
    print(outputs)
    print(inputs)
    tf.saved_model.simple_save(
            sess,
            export_path,
            inputs=inputs,
            outputs=outputs)


    return export_path

if __name__ == "__main__":
    model_path=os.path.join(save_model_dir, "tfserving", '{}'.format(model_type))
    onnx_path = os.path.join(save_model_dir, "onnx", model_type+'.onnx')
    export_path = convert_to_saved_model(model, model_path)
    print(f"python -m --saved-model  {export_path}  --output  {onnx_path}  --inputs Input-Token:0,Input-Segment:0 --outputs cross_entropy_1/Identity:0   --opset 11")
  
    """
    docker run -p 8511:8501 -p 8510:8500  \
    --gpus '"device=0"' \
    --name tfserving_surg \
    --mount type=bind,source=/home/qyc/generateSurg/save/tfserving/new_operation,target=/models/surg_model \
    -e MODEL_NAME=surg_model \
    tensorflow/serving
    """
    #请求tfserving
    #http://192.168.0.181:8501/v1/models/surg_model
    #pip install tensorflow-serving-api==1.14.0
    import grpc
    from tensorflow_serving.apis import model_service_pb2_grpc, get_model_status_pb2

    def check_tfserving_health(host="localhost", port=8500):
        """检查 TF Serving 健康状态"""
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        
        request = get_model_status_pb2.GetModelStatusRequest()
        request.model_spec.name = "surg_model"
        
        try:
            response = stub.GetModelStatus(request, timeout=5)
            print("✅ TF Serving 正常")
            print(response)
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    # check_tfserving_health(host="localhost", port=8500)
    check_tfserving_health(host="192.168.0.180", port=8510)

```