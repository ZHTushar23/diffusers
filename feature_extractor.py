import torch
import torchvision.models as models
import torchvision.transforms as transforms


def get_features(input_image,device="cpu"):
    # Load pre-trained ResNet-50 model
    # resnet50 = models.resnet50(weights="IMAGENET1K_V1")
    resnet50 = models.resnet18(weights="IMAGENET1K_V1").to(device)

    # Remove the fully connected layer (classification layer)
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

    # Set the model to evaluation mode
    resnet50.eval()

    # Define a preprocessing function for your input images
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)
        # image = image.unsqueeze(0)  # Add batch dimension
        return image

    # Example usage
    input_image = preprocess_image(input_image)

    # Forward pass through the ResNet-50 model
    with torch.no_grad():
        output_features = resnet50(input_image)
    return output_features

if __name__=="__main__":
    original_tensor = torch.randn(1,3, 72, 72)
    output_features=get_features(original_tensor)
    # The output_features tensor now contains the extracted features
    print("Shape of the extracted features:", output_features.shape)
    print(original_tensor.dtype)
    print(output_features.dtype)



    # Assuming you have a tensor of size [1, 512, 1, 1]
    tensor_to_squeeze = torch.randn(1, 512, 1, 1)

    # Squeeze the tensor
    squeezed_tensor = tensor_to_squeeze.squeeze()

    # Check the size of the squeezed tensor
    print("Original Tensor Size:", tensor_to_squeeze.size())
    print("Squeezed Tensor Size:", squeezed_tensor.size())
