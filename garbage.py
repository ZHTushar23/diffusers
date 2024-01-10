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

custom_dataset = NasaDataset(root_dir=dataset_dir,ref_scale=False)
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
# Extract indices from the train_data and test_data
train_indices = train_data.indices
test_indices = test_data.indices

# Print or use the indices as needed
# print("Training set indices:", train_indices)
print("Test set indices:", test_indices)
