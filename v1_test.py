from diffusers import DDPMPipeline, UNet2DModel, DiffusionPipeline
from diffusers import DDPMScheduler
from v1_utilities import TrainingConfig
from evaluation import evaluate
import os

config = TrainingConfig()
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
model = UNet2DModel(
    sample_size=72,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        # "DownBlock2D",
        # "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
        "UpBlock2D",
    ),
)

# load the model
generator = DiffusionPipeline.from_pretrained("ddpm-cot-72").to("cuda")
image = generator().images[0]
print(type(image))
print(image.shape)
test_dir = os.path.join(config.output_dir, "samples")
image.save(f"{test_dir}/newsample.png")
# pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
# evaluate(config, 1, pipeline)