# from diffusers import DDPMPipeline, UNet2DModel, DiffusionPipeline
from diffusers import DDPMScheduler
import os
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T

from diffusion_mini_cond_dummy import DiffusionMiniCondD
from visualization import *
from v4_config import TrainingConfig
from v4_dataloader import NasaDataset,rescale
import v3_pipeline 
from feature_extractor import get_features
from visualization import *

DEVICE="cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# 0. load config
config = TrainingConfig()

# 1. load noise scheduler. It generates and removes noise. 
noise_scheduler = DDPMScheduler(num_train_timesteps=100)

# 2. load model
model = DiffusionMiniCondD(in_channels=1,interim_channels=32,out_channels=1)
# model_saved_path = os.path.join(config.output_dir,"model.pth")
# print(model_saved_path)
# model.load_state_dict(torch.load(model_saved_path,map_location=torch.device('cpu')))

# 3. load test dataset 
config.dataset_name = "cotF2"
dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"
transform = T.Compose([
            # T.Resize(256),
            # T.CenterCrop(224),
            # T.ToTensor(),
            T.Normalize(mean=[0.1112], std=[0.1847]),
        ])
test_data = NasaDataset(root_dir=dataset_dir,mode="train",transform_cot=transform)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

mse_loss = []
dir_name = config.output_dir+"/samples"
try:
    os.makedirs(dir_name)
except FileExistsError:
    print("folder already exists")

for i in range(len(test_dataloader.dataset)):
    data = test_dataloader.dataset[i]
    p_num =  i
    # get the data
    target_image, input_image, context = data['reflectance'],data['cot'],data['angles']
    # break

    input_image = torch.unsqueeze(input_image,0)
    context = torch.unsqueeze(context,0)

    input_image= input_image.to(dtype=torch.float32)

    # get features using pretrained resnet18 model. 
    cot_context = get_features(input_image)
    # del input_image
    cot_context = cot_context.squeeze()

    # ## SAMPLER
    sampler = "ddpm"
    num_inference_steps = 1000
    seed = 42

    output_image = v3_pipeline.generate(
        prompt=context,
        input_image=cot_context,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        model=model,
        device=DEVICE
    )
    print(output_image.shape)
    # Combine the input image and the output image into a single image.
    # target_image = rescale(target_image,(-1, 1),(0, 2.1272))
    target_image = target_image.numpy()

    # # Plot COT
    # p_num=100
    
    fname = dir_name+"/full_profile_jet_norm_rad066_pred_%01d.png"%(p_num)
    plot_cot2(cot=output_image[:,:,0],title="Pred Radiance 0.66um",fname=fname,use_log=False,limit=[0,2])

    fname = dir_name+"/full_profile_jet_norm_rad066_%01d.png"%(p_num)
    plot_cot2(cot=target_image[0,:,:],title="True Radiance 0.66um",fname=fname,use_log=False,limit=[0,2])

    fname = dir_name+"/full_profile_jet_norm_cot_%01d.png"%(p_num)
    plot_cot2(cot=input_image[0,0,:,:],title="True COT",fname=fname,use_log=False,limit=[0,7])

    mse_loss.append(np.average((target_image[0,:,:]-output_image[:,:,0])**2))
    if i==1:
        break

print("Avergae MSE: ", np.average(mse_loss))    