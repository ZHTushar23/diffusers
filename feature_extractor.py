import torch
import torchvision.models as models
import torchvision.transforms as transforms
from encoder2decoder import E2D
import torch.nn as nn

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


def get_features_e2d(input_image,device="cpu"):
    saved_model_path = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/v60_saved_model/e2d/e2d_fold__20231227_111633.pth"
    # Load pre-trained Encoder2Decoder model
    model = E2D(n_channels=1,n_dim=32)
    model.load_state_dict(torch.load(saved_model_path,map_location=torch.device('cpu')))
    model.to(device)

    # Remove the decoder
    model = torch.nn.Sequential(*(list(model.children())[:-5]))
    m = nn.MaxPool2d(9, stride=1)
    # # Define the layer from which you want to extract features
    # target_layer = model.down3

    # # Placeholder to store the extracted features
    # features = None

    # # Define a hook function to be called when forward pass reaches the target layer
    # def hook(module, input, output):
    #     global features
    #     features = output

    # Register the hook to the target layer
    # hook_handle = target_layer.register_forward_hook(hook)
    # Set the model to evaluation mode
    model.eval()


    # Forward pass through the ResNet-50 model
    with torch.no_grad():
        output_features = model(input_image)
        output_features = m(output_features)
    return output_features

def get_features_e2dCNN(input_image,device="cpu"):
    saved_model_path = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/v60_saved_model/e2dCNN/e2d_fold__20231227_111633.pth"
    # Load pre-trained Encoder2Decoder model
    model = E2D(n_channels=1,n_dim=32)
    model.load_state_dict(torch.load(saved_model_path,map_location=torch.device('cpu')))
    model.to(device)

    # Remove the decoder
    model = torch.nn.Sequential(*(list(model.children())[:-5]))
    m = nn.MaxPool2d(9, stride=1)
    # # Define the layer from which you want to extract features
    # target_layer = model.down3

    # # Placeholder to store the extracted features
    # features = None

    # # Define a hook function to be called when forward pass reaches the target layer
    # def hook(module, input, output):
    #     global features
    #     features = output

    # Register the hook to the target layer
    # hook_handle = target_layer.register_forward_hook(hook)
    # Set the model to evaluation mode
    model.eval()


    # Forward pass through the ResNet-50 model
    with torch.no_grad():
        output_features = model(input_image)
        output_features = m(output_features)
    return output_features

if __name__=="__main__":
    original_tensor = torch.randn(1,3, 72, 72)
    output_features=get_features(original_tensor)
    # output_features=get_features_e2d(original_tensor)
    # # The output_features tensor now contains the extracted features
    print("Shape of the extracted features:", output_features.shape)
    print(original_tensor.dtype)
    print(output_features.dtype)



    # # Assuming you have a tensor of size [1, 512, 1, 1]
    # tensor_to_squeeze = torch.randn(1, 512, 1, 1)

    # # Squeeze the tensor
    # squeezed_tensor = tensor_to_squeeze.squeeze()

    # # Check the size of the squeezed tensor
    # print("Original Tensor Size:", tensor_to_squeeze.size())
    # print("Squeezed Tensor Size:", squeezed_tensor.size())
