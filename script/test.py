#!/usr/bin/env python3

from diffusers import Diffusion

# 加载模型
model = Diffusion.from_pretrained("path/to/model.safetensors")

# 生成图像
image = model.generate(input_image)

# 保存图像
image.save("output.png")