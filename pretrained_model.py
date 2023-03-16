import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

def load_pretrained_model(input_channels=1, num_classes=2, freeze_base_model=True):
    # Load the pre-trained 3D ResNet model
    model = r3d_18(pretrained=True)

    # Modify the first layer to accept the number of input channels in your dataset
    model.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

    # Freeze the base model if specified
    if freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False

    # Modify the last layer to have the same number of output classes as your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

if __name__ == '__main__':
    model = load_pretrained_model()
    print('Pre-trained model loaded and modified:')
    print(model)
