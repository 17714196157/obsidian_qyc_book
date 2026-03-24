[ltx](https://ltx.io/ltx-desktop)


**国内镜像下载（ModelScope）：**
```bash
# 下载主模型
modelscope download --model AI-ModelScope/LTX-Video ltx-video-2b-v0.9.safetensors --local_dir ./models/checkpoints/

# 下载文本编码器
modelscope download --model Comfy-Org/stable-diffusion-3.5-fp8 text_encoders/t5xxl_fp16.safetensors --local_dir ./models/text_encoder/

ComfyUI/
├── models/
│   ├── checkpoints/          # 主模型放这里
│   │   └── ltx-2-19b-dev.safetensors
│   ├── loras/                # LoRA文件
│   │   └── ltx-2-19b-distilled-lora-384.safetensors
│   ├── latent_upscale_models/ # 放大器模型
│   │   ├── ltx-2-spatial-upscaler-x2-1.0.safetensors
│   │   └── ltx-2-temporal-upscaler-x2-1.0.safetensors
│   ├── text_encoders/         # 文本编码器
│   │   └── gemma-3-12b-it-qat-q4_0-unquantized/
│   └── unet/                  # GGUF模型放这里（如使用）
│       └── ltx-2-Q4_K_M.gguf
```
1. 启动 ComfyUI（`python main.py` 或双击 bat 文件）
2. 访问 [http://127.0.0.1:8188](http://127.0.0.1:8188/)
3. 从官方仓库下载工作流模板：[https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows](https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows)
    - `LTX-2_T2V_Full.json` - 文生视频完整版
    - `LTX-2_I2V_Full.json` - 图生视频完整版
    - `LTX-2_T2V_Distilled.json` - 快速蒸馏版
