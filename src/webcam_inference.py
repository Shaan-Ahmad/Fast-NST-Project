import torch
import torch.nn as nn
import numpy as np
import cv2 
import time

from PIL import Image
from torchvision import transforms

# Import the Generator model we trained
from models.generator import TransformerNet 

# Configuration 
MODEL_PATH = 'saved_models/starry_night_epoch_1.pth' 
# Webcam resolution 
TARGET_WIDTH = 640 
TARGET_HEIGHT = 480 

# Main Inference Function 
def stylize_video():
    # Device Setup 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("Warning: CUDA not found. Running on CPU, which may be slow for real-time.")
    
    # Model Loading
    print(f"Loading model from: {MODEL_PATH}")
    # Initialize the architecture
    transformer = TransformerNet().to(device)
    
    # Load the trained state_dict (weights)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Fix a common issue where InstanceNorm layers cause problems when loading
    for k in list(state_dict.keys()):
        if k.endswith('.running_mean') or k.endswith('.running_var'):
            del state_dict[k]
    
    transformer.load_state_dict(state_dict, strict=False)
    transformer.eval()

    # Video Capture and Image Pipeline Setup
    cap = cv2.VideoCapture(0) # 0 is typically the default webcam
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Check camera index or permissions.")
        
    # PyTorch frame transformation (input)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # --- Real-Time Loop ---
    print("Starting real-time stylization.")
    
    fps_start_time = time.time()
    frame_count = 0
    
    while True:
        # Capture Frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Resize frame for consistent GPU processing size 
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # Pre-processing (CPU to GPU)
        # Convert BGR frame to RGB 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply PyTorch transform and move tensor to GPU
        content_image = content_transform(rgb_frame).unsqueeze(0).to(device)
        
        # GPU Inference
        with torch.no_grad(): # Crucial for speed and memory efficiency
            output_tensor = transformer(content_image)

        # Post-processing (GPU to CPU)
        output_image = output_tensor.cpu().clone().squeeze(0).clamp(0, 1).numpy()
        
        # Rearrange dimensions and convert from RGB back to BGR for OpenCV display
        output_image = output_image.transpose(1, 2, 0) 
        output_image = cv2.cvtColor(output_image * 255, cv2.COLOR_RGB2BGR).astype(np.uint8)

        # Display and FPS calculation
        
        # Calculate FPS
        frame_count += 1
        fps_current_time = time.time()
        fps = frame_count / (fps_current_time - fps_start_time)
        
        # Draw FPS on the output window
        cv2.putText(output_image, f"FPS: {fps:.2f} (RTX 3050)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        cv2.imshow("Real-Time Stylization", output_image)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == '__main__':
    stylize_video()