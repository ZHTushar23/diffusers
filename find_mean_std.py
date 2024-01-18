# Import Libraries
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np
from v3_dataloader import NasaDataset
import torchvision.transforms as T
torch.manual_seed(0)
'''
        Dataset types:
        1. 'cloud_25'
        2. 'cloud_50'
        3. 'cloud_75'
        4. 'cv_dataset'
'''
def get_mean_and_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for batch, sample in enumerate (loader):
        data = sample['cot']
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# Dataset Name
# dataset_name = 'cv_dataset'
# # Data source csv file name and directory
# root_dir = "/home/local/AD/ztushar1/"
# root_data_dir = root_dir+"LES_SingleView2/"

# for k in range (5):
    # csv_file = root_data_dir+"CV/"+"Train_fold_%01d.csv"%(k)
    # train_data = NasaDataset(
    #                         root_dir=root_data_dir,
    #                         csv_file = csv_file,
    #                         dataset_name = dataset_name,
    #                         fold = k, transform=True
    # )
    # loader = DataLoader(train_data, batch_size=10)
dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"
transform = T.Compose([
            # T.Resize(256),
            # T.CenterCrop(224),
            # T.ToTensor(),
            T.Normalize(mean=[0.1112], std=[0.1847]),
        ])

train_data = NasaDataset(root_dir=dataset_dir,mode="train",transform_cot=transform)
loader = DataLoader(train_data, batch_size=10,shuffle=False)
mean, std = get_mean_and_std(loader)


print("Dataset Mean: ",mean)
print("Dataset Std: ",std)
print("Done!")