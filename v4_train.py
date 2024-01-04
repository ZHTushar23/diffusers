#https://huggingface.co/docs/diffusers/v0.24.0/en/tutorials/basic_training#training-configuration

# import libraries
from v4_dataloader import NasaDataset
import torchvision.transforms as T
from v4_config import *
from torch.utils.data import  random_split, DataLoader
import torch
import matplotlib.pyplot as plt
# from diffusers import UNet2DModel
from diffusion_mini_cond_e2d import DiffusionMiniCondD
from v3_train_utils import train_loop
import os
local_rank = int(os.environ["LOCAL_RANK"])
# 1. training configuraion
config = TrainingConfig()


# 2. load the dataset
config.dataset_name = "cotF2"
dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"
transform = T.Compose([
            # T.Resize(256),
            # T.CenterCrop(224),
            # T.ToTensor(),
            T.Normalize(mean=[0.6096], std=[1.0741]),
        ])
custom_dataset = NasaDataset(root_dir=dataset_dir,transform_cot=transform)
# Create a separate generator for the random split
split_generator = torch.Generator()
split_generator.manual_seed(13)  # You can choose any seed value

# Define the sizes for train, validation, and test sets
total_size = len(custom_dataset)
test_size = int(0.2 * total_size)
# Use random_split to split the dataset
train_data, test_data = random_split(
    custom_dataset, [total_size - test_size, test_size], generator=split_generator
)

train_dataloader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)


# 3. Create a UNet2DModel
model = DiffusionMiniCondD(in_channels=1,interim_channels=32,out_channels=1)
# # Check if the model initialization is okay.
# sample_image = train_data[0]["cot"].unsqueeze(0)
# print("Input shape:", sample_image.shape)

# print("Output shape:", model(sample_image, timestep=0).sample.shape)
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of parameters: {:,}".format(num_params))


# 4. Create a scheduler
'''
The scheduler behaves differently depending on whether you are using the model for training or inference. 
During inference, the scheduler generates image from the noise. 
During training, the scheduler takes a model output - or a sample - from a specific point 
in the diffusion process and applies noise to the image according to a noise schedule 
and an update rule.
'''

from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# # 5. loss objectives
# import torch.nn.functional as F
# sample_image = train_data[0]["cot"].unsqueeze(0)
# noise = torch.randn(sample_image.shape)
# timesteps = torch.LongTensor([50])
# noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
# noise_pred = model(noisy_image, timesteps).sample
# loss = F.mse_loss(noise_pred, noise)
# print(loss.item())

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