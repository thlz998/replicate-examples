#!/bin/bash

cog predict \
-i prompt="monkey scuba diving" \
-i qr_code_content="殷迦南殷迦南殷迦南" \
-i negative_prompt="ugly, disfigured, low quality, blurry, nsfw" \
-i num_inference_steps=100 \
-i guidance_scale=30 \
-i seed=-1 \
-i batch_size=1 \
-i strength=1 \
-i controlnet_conditioning_scale=2