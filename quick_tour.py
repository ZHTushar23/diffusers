from accelerate import Accelerator

accelerator = Accelerator()
# check the scheduler
import torch
from PIL import Image
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# # check scheduler
# sample_image = dataset[0]["images"].unsqueeze(0)
# noise = torch.randn(sample_image.shape)
# timesteps = torch.LongTensor([50])
# noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

# a = Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

# fig, axs = plt.subplots(1, 1, figsize=(4, 4))
# axs.imshow(a)
# axs.set_axis_off()
# plt.savefig("demo2.png")
# plt.close()