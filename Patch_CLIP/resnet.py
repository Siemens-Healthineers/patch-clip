import torch
import torch.nn as nn

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (one residual block)
        self.residual_block = self._make_residual_block(64, 512)

        # Average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((512, 1))

    def _make_residual_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.residual_block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class MLPDownsampler(nn.Module):
    def __init__(self, in_channels: int, input_resolution: int, out_resolution: int):
        super(SimpleMLP, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 1024 * 1024, 3 * 512 * 512)
        self.fc2 = nn.Linear(3* 512 * 512, 3 * 224 * 224)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = x.view(-1, 3, 224, 224)
        return x

class SimpleDownsample(nn.Module):
    def __init__(self, in_channels: int, input_resolution: int, out_resolution: int):
        super(SimpleDownsample, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Additional convolutional layers for downsampling
        self.conv_downsample = nn.Sequential(
            #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(128),
            #nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            #nn.Sigmoid()  # To ensure output values are in [0, 1] range
        )

    def forward(self, x):
        x = self.conv1(x).to(x.dtype) #when training make sure its not in half()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_downsample(x)
        return x

do_eval = False
if do_eval:
    # Create an instance of the modified SimpleResNetDownsample model
    model_resnet_downsample = SimpleDownsample()

    # Generate a random input tensor with the shape 3x1024x1024
    input_tensor = torch.rand(1, 3, 1024, 1024)

    # Pass the input through the modified ResNet model
    output_tensor_resnet_downsample = model_resnet_downsample(input_tensor)

    # Print the shape of the output tensor
    print("Output tensor shape (modified ResNet with downsample):", output_tensor_resnet_downsample.shape)

