Fun-ASR-Nano-2512 | 阿里最新开源的Fun-ASR-Nano-2512体验 | Fun-ASR-Nano-2512方言测试体验

测试体验地址：https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano 
项目开源地址：https://github.com/FunAudioLLM/Fun-ASR 



区分说话人语音识别开源啦 | 语音识别 | 区分说话人 | 基于FunASR开发的语音识别接口 | 可内网部署的区分说话人语音转写服务 
开源地址：https://github.com/lukeewin/FunASR_API.git 

1） 创建mysql
```
docker run -d \
  --name mysql-local \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=123456 \
  -e MYSQL_DATABASE=testdb \
  -v ~/mysql-data:/var/lib/mysql \
  --restart=unless-stopped \
  mysql:8.0 \
  --character-set-server=utf8mb4 \
  --collation-server=utf8mb4_unicode_ci

```

2） 创建数据库
```
docker exec -i mysql-local mysql -uroot -p123456 -e "CREATE DATABASE IF NOT EXISTS funasr_api CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```
执行 funasr_api.sql 文件 创建表

3）
模型下载
```
modelscope download --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
m
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
m
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
m
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common
```

4） 测试
 curl -F "file=@output.wav" http://192.168.0.181:8509/trans/file



5） asr.py  给用streamlit 做一个界面， 上传录音文件， 展示接口返回的识别文本内容

![[AI应用/AI语音/assets/阿里最新开源的Fun-ASR 语音识别/53db1df459aa12792db638cee53cd439_MD5.png]]