import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipe.unet.load_attn_procs("saved-model", weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion("saved-model", weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a sofa",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat2.png")