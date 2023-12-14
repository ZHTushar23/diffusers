from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
import torch
# image transform
'''
resize, horizontal flip, 
normalize: Normalize is important to rescale the pixel values into a [-1, 1] range, 
which is what the model expects.
'''


if __name__=="__main__":

    # import libraries
    from datasets import load_dataset
    from utilities import *
    # training configuraion
    config = TrainingConfig()

    # load the dataset
    config.dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(config.dataset_name, split="train")

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

    # fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    # for i, image in enumerate(dataset[:4]["image"]):
    #     axs[i].imshow(image)
    #     axs[i].set_axis_off()
    # # fig.show()
    # plt.savefig("demo3.png")
    # plt.close()
    dataset.set_transform(transform)
    loader = DataLoader(dataset, batch_size=2,shuffle=False)
    for step, batch in enumerate(loader):
        clean_images = batch["images"]
        print(clean_images.shape)
        print(torch.min(clean_images),torch.max(clean_images))
        break
