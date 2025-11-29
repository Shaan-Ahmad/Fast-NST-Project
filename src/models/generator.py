import torch
import torch.nn as nn
import torch.nn.functional as F

#Helper Module: Convolutional Layer with Instance Normalization
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        return self.conv(x)

#Helper Module: Residual Block (for content preservation)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity 
        return out

#Helper Module: Deconvolution (for Upsampling) 
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride), 
            nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        return self.conv(x)


#The Main Generator Network (TransformerNet)
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # Initial Layers (Encoder)
        self.initial_layers = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2), 
            nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=2), 
            nn.ReLU()
        )
        
        # Residual Blocks (Transformation Core)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # Upsampling Layers (Decoder)
        self.upsample_layers = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2), 
            nn.ReLU(),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2), 
            nn.ReLU(),
            # Output Layer: maps to 3 RGB channels
            ConvLayer(32, 3, kernel_size=9, stride=1) 
        )

    def forward(self, X):
        X = self.initial_layers(X)
        X = self.res_blocks(X)
        X = self.upsample_layers(X)
        
        # Tanh maps output to [-1, 1]. We scale and clip the output to [0, 1] for images.
        return torch.sigmoid(X)