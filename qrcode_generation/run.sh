#!/bin/bash

cog predict \
-i prompt="monkey scuba diving" \
-i qr_code_content="https://dl-1257240317.cos.ap-guangzhou.myqcloud.com/k0a1a/test-qrcode.png" \
-i negative_prompt="ugly, disfigured, low quality, blurry, nsfw" \
-i num_inference_steps=100 \
-i guidance_scale=30 \
-i seed=-1 \
-i batch_size=1 \
-i strength=1 \
-i controlnet_conditioning_scale=2