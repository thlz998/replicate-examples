from typing import List

import torch
import time
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, EulerDiscreteScheduler


CACHE_DIR = "weights-cache"

def resize_for_condition_image(input_image, resolution: int):
    from PIL.Image import LANCZOS

    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=LANCZOS)
    return img


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # torch.backends.cuda.matmul.allow_tf32 = True
        print("开始时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        controlnet_qr_pattern = ControlNetModel.from_pretrained(
        "Nacholmo/controlnet-qr-pattern",  
        torch_dtype=torch.float16,
        force_download=False,
        cache_dir=CACHE_DIR).to("cuda")

        controlnet_qr_monster = ControlNetModel.from_pretrained(
                "monster-labs/control_v1p_sd15_qrcode_monster", 
                torch_dtype=torch.float16,
                force_download=False,
                cache_dir=CACHE_DIR).to("cuda")

        self.controlnet = [controlnet_qr_monster, controlnet_qr_pattern]
        # self.pipe.enable_xformers_memory_efficient_attention()

    def generate_qrcode(self, qr_url: str):
        # 从URL中下载二维码
        import requests
        from io import BytesIO
        from PIL import Image
        response = requests.get(qr_url)
        qrcode_image = Image.open(BytesIO(response.content))
        qrcode_image = resize_for_condition_image(qrcode_image, 768)
        return qrcode_image

    # Define the arguments and types the model takes as input
    def predict(
        self,
        prompt: str = Input(description="QR Code Prompt"),
        # qr_code_content: str = Input(description="二维码内容"),
        qr_code_content: str = Input(description="二维码URL", default="https://dl-1257240317.cos.ap-guangzhou.myqcloud.com/k0a1a/test-qrcode.png"),
        negative_prompt: str = Input(
            description="反向提示词",
            default="ugly, disfigured, low quality, blurry, nsfw",
        ),
        num_inference_steps: int = Input(description="Number of diffusion steps", ge=20, le=100, default=40),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=-1),
        batch_size: int = Input(description="Batch size for this prediction", ge=1, le=4, default=1),
        # controlnet_conditioning_scale: float = Input(
        #     description="The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original unet.",
        #     ge=1.0,
        #     le=2.0,
        #     default=1.5,
        # ),
    ) -> List[Path]:
        seed = torch.randint(0, 2**32, (1,)).item() if seed == -1 else seed
        qrcode_image = self.generate_qrcode(qr_code_content)
        control_image = [qrcode_image] * batch_size
        
        model_key_base_safetensors = "./weights-cache/models/revAnimated_v122.safetensors"
        pipe = StableDiffusionPipeline.from_single_file(
                model_key_base_safetensors,
                torch_dtype=torch.float16,
                cache_dir=CACHE_DIR).to("cuda")
        components = pipe.components

        components["controlnet"] = self.controlnet
        pipe = StableDiffusionControlNetPipeline(**components).to("cuda")

        lora_model_file = './weights-cache/Lora/blindbox_v1_mix.safetensors'
        pipe.load_lora_weights(lora_model_file, cache_dir=CACHE_DIR)
        pipe = pipe.to("cuda", torch.float16)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        guidance_scale = 7
        num_inference_steps = 35

        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
        pipe.scheduler.set_timesteps(num_inference_steps)

        out = pipe(
            width=512,
            height=512,
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            image=control_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=[1.0, 1.0],
            control_guidance_start=[0, 0],
            control_guidance_end=[1, 1]
        )
        print("结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for i, image in enumerate(out.images):
            fname = f"output-{i}.png"
            image.save(fname)
            yield Path(fname)
