from diffusers import DDPMPipeline, AutoPipelineForImage2Image
from diffusers.utils import make_image_grid
import os
import torch

def evaluate(config, epoch, pipeline,test_dataloader):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    for step, batch in enumerate(test_dataloader):
        init_image,context_embedding = batch["cot"],batch["angles"]
        images = pipeline(image=init_image, prompt_embeds=context_embedding
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
        ).images

        # Make a grid out of the images
        image_grid = make_image_grid(images, rows=4, cols=4)

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")