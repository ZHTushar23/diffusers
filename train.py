#https://huggingface.co/docs/diffusers/v0.24.0/en/tutorials/basic_training#training-configuration

# import libraries
from datasets import load_dataset
from utilities import *
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from diffusers import UNet2DModel
from train_utils import train_loop
import os
local_rank = int(os.environ["LOCAL_RANK"])
# 1. training configuraion
config = TrainingConfig()


# 2. load the dataset
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")

# data transformation
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

print(len(train_dataloader))



# 3. Create a UNet2DModel
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# # Check if the model initialization is okay.
# sample_image = dataset[0]["images"].unsqueeze(0)
# print("Input shape:", sample_image.shape)

# print("Output shape:", model(sample_image, timestep=0).sample.shape)



# 4. Create a scheduler
'''
The scheduler behaves differently depending on whether you are using the model for training or inference. 
During inference, the scheduler generates image from the noise. 
During training, the scheduler takes a model output - or a sample - from a specific point 
in the diffusion process and applies noise to the image according to a noise schedule 
and an update rule.
'''
import torch
from PIL import Image
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# # 5. loss objectives
# import torch.nn.functional as F
# sample_image = dataset[0]["images"].unsqueeze(0)
# noise = torch.randn(sample_image.shape)
# timesteps = torch.LongTensor([50])
# noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
# noise_pred = model(noisy_image, timesteps).sample
# loss = F.mse_loss(noise_pred, noise)
# # print(loss.item())

# 6. optimizer
from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


# 7. train the model
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)