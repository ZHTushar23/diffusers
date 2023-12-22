import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import matplotlib.pyplot as plt

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()

# 2. Load an image to pass to the pipeline:
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

# 3. Pass a prompt and image to the pipeline to generate an image:
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
image.save("im2img_cat.png")
g = make_image_grid([init_image, image], rows=1, cols=2)
