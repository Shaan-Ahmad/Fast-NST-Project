import torch
import torch.nn as nn
from torchvision import models

class VGG16_LossNetwork(nn.Module):
    # Mapping VGG module indices to common layer names
    VGG_LAYERS = {
        '0': 'relu1_1',  
        '5': 'relu2_1',  
        '10': 'relu3_1', 
        '17': 'relu4_1',
        '24': 'relu5_1'
    }

    def __init__(self, style_layers=None, content_layer=None):
        super(VGG16_LossNetwork, self).__init__()
        
        # Load pre-trained VGG16 features and move to GPU
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.cuda().eval()
        
        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.content_layer = content_layer if content_layer else 'relu3_1'
        self.style_layers = style_layers if style_layers else ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        
        self.layer_indices = {name: int(idx) for idx, name in self.VGG_LAYERS.items()}

    def forward(self, x):
        features = {}
        # Iterate through the VGG layers
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            
            layer_name = self.VGG_LAYERS.get(str(i))
            if layer_name:
                features[layer_name] = x
            
            # Optimization
            required_layers = [self.content_layer] + self.style_layers
            if all(layer in features for layer in required_layers):
                break
                
        return features

    # Static Method for VGG Input Pre-processing
    @staticmethod
    def normalize_batch(batch):
        """Standard ImageNet normalization for VGG input"""
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        # Input batch is assumed to be scaled to [0, 1]
        return (batch - mean) / std