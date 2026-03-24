参考地址：https://github.com/index-tts/index-tts

安装：
```
pip install -U uv
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
🌐 Web Demo
uv run webui.py

```


![[AI应用/AI语音/assets/IndexTTS/982df248b2f04eb4ffee3f18bbda4909_MD5.png]]
下载api服务 https://cnb.cool/fangwaii/index-tts-v2/-/blob/main/api.py
uv run api.py


也可以用CNB云服务 https://cnb.cool/fangwaii/index-tts-v2
![[AI应用/AI语音/assets/IndexTTS/f6432caefd64950b8f4ff2bccdcc26e1_MD5.png]]


![[AI应用/AI语音/assets/IndexTTS/02ed27be0b4263451af3e721e272652b_MD5.png]]
CNB云服务地址：

[[AI应用/AI语音/assets/IndexTTS/2236593c185e8b868664f5d07b881ea8_MD5.png|Open: file-20260324191248315.png]]
![[AI应用/AI语音/assets/IndexTTS/2236593c185e8b868664f5d07b881ea8_MD5.png]]


代码请求：
```python
    from indextts.infer import IndexTTS
    tts = IndexTTS(model_dir="/home/qyc/index-tts/checkpoints",cfg_path="/home/qyc/index-tts/checkpoints/config.yaml")
    voice = "/home/qyc/index-tts/examples/voice_07.wav"
    text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
    tts.infer(voice, text, 'gen.wav')
```