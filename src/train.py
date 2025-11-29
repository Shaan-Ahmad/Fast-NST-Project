import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os 
from PIL import Image

from models.generator import TransformerNet 
from models.loss_network import VGG16_LossNetwork 

# Helper Function: Gram Matrix (for Style Loss)
def gram_matrix(input):
    # input is a BxCxHxW tensor 
    a, b, c, d = input.size()  
    
    # Reshape features to a BxCx(H*W) matrix 
    features = input.view(a, b, c * d) 
    G = torch.matmul(features, features.transpose(1, 2)) 
    return G.div(c * d)

# Hyperparameters and Configuration 

CONTENT_WEIGHT = 1.0e5 
STYLE_WEIGHT = 1.0e10
TV_WEIGHT = 1.0e-7 
LR = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 4
IMAGE_SIZE = 256
DATASET_PATH = 'data/mscoco' 
STYLE_IMAGE_PATH = 'styles/starry_night.jpg' 

def train():
    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading 
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),  
    ])

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}. Please ensure your images are in 'data/mscoco/train/all_images'.")

    # Load images 
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Check if the loader found any images
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in {DATASET_PATH}. Check your folder structure!")

    # Model Initialization 
    transformer = TransformerNet().to(device)
    loss_network = VGG16_LossNetwork().to(device)
    optimizer = optim.Adam(transformer.parameters(), LR)
    mse_loss = nn.MSELoss().to(device) 

    # Pre-calculate Style Features  
    if not os.path.exists(STYLE_IMAGE_PATH):
         raise FileNotFoundError(f"Style image not found at: {STYLE_IMAGE_PATH}")
         
    style_img = transform(Image.open(STYLE_IMAGE_PATH).convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad(): 
        style_features = loss_network(VGG16_LossNetwork.normalize_batch(style_img))
        style_grams = {layer: gram_matrix(features) for layer, features in style_features.items()}

    # Training Loop 
    for epoch in range(NUM_EPOCHS):
        transformer.train() 
        start_time = time.time()
        
        for batch_id, (content_batch, _) in enumerate(train_loader):
            n_batch = len(content_batch)
            content_batch = content_batch.to(device)
            
            # Forward Pass & Loss Calculation
            optimizer.zero_grad()
            generated_batch = transformer(content_batch)
            
            # VGG features for content and generated image
            content_features = loss_network(VGG16_LossNetwork.normalize_batch(content_batch))
            generated_features = loss_network(VGG16_LossNetwork.normalize_batch(generated_batch))

            # Content Loss
            content_loss = mse_loss(
                generated_features[loss_network.content_layer],
                content_features[loss_network.content_layer]
            )

            # Style Loss
            style_loss = 0.0
            for layer in loss_network.style_layers:
                generated_gram = gram_matrix(generated_features[layer])
                style_loss += mse_loss(generated_gram, style_grams[layer])
            
            # Total Variation Loss (Smoothing)
            tv_loss = (torch.sum(torch.abs(generated_batch[:, :, :, :-1] - generated_batch[:, :, :, 1:])) + 
                       torch.sum(torch.abs(generated_batch[:, :, :-1, :] - generated_batch[:, :, 1:, :])))

            # Total Loss (Weighted sum)
            total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss

            # Optimization
            total_loss.backward()
            optimizer.step()

            # Logging 
            if batch_id % 50 == 0:
                print(f"Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_id} | Time: {time.time() - start_time:.2f}s | "
                      f"Total Loss: {total_loss.item():.4f} | Content: {content_loss.item():.2f} | Style: {style_loss.item():.2f}")
                start_time = time.time()

        # Save the model checkpoint 
        model_name = STYLE_IMAGE_PATH.split('/')[-1].split('.')[0]
        torch.save(transformer.state_dict(), f'saved_models/{model_name}_epoch_{epoch}.pth')
        print(f"Model saved to saved_models/{model_name}_epoch_{epoch}.pth")


if __name__ == '__main__':
    train()