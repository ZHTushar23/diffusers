# import libraries
from v6_dataloader import NasaDataset
import torchvision.transforms as T
from v6_config import *
from torch.utils.data import  random_split, DataLoader
import torch

# 1. training configuraion
config = TrainingConfig()
# 2. load the dataset
config.dataset_name = "cotF2"
dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"

train_data = NasaDataset(root_dir=dataset_dir,mode="train",ref_scale=False)


# Print or use the indices as needed
# print("Training set indices:", train_indices)
# print("Test set indices:", test_indices)
train_dataloader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
for idx, batch in enumerate(train_dataloader):
    print( type(batch))
    clean_images, angle_context, cot_data = batch["reflectance"],batch["angles"],batch["cot"]
    print(clean_images.shape, angle_context.shape, cot_data.shape)
    break