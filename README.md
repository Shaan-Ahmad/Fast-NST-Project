# Real-Time Fast Neural Style Transfer (Fast-NST)

**Project Status:** COMPLETE | **Hardware:** NVIDIA RTX 3050 | **Framework:** PyTorch

---

## What is This Project?

This project lets you take a photograph (the **style image**) and apply its artistic look to a live webcam video in real-time.

It uses a deep learning model to instantly paint a video feed with the textures and colors of a famous painting, like Van Gogh's *Starry Night*.

### Key Features:
* **Live Stylization:** Applies style to your webcam video instantly (no lag).
* **Custom Models:** Trained to apply the unique style of *Starry Night*.
* **Optimized for GPU:** Built with PyTorch for fast performance on NVIDIA GPUs.

---

## Setup and Installation

### Prerequisites
You need a system with Python 3, Git, and a working **NVIDIA GPU** with the correct drivers (CUDA).

### Steps to Run
1.  **Clone the Project:**
    ```bash
    git clone [https://github.com/Shaan-Ahmad/Fast-NST-Project.git](https://github.com/Shaan-Ahmad/Fast-NST-Project.git)
    cd Fast-NST-Project
    ```

2.  **Create and Activate Environment:**
    ```bash
    python3 -m venv venv_style_transfer
    source venv_style_transfer/bin/activate
    ```

3.  **Install Required Libraries:**
    * *Note: If you are using a different CUDA version, adjust the command below.*
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    pip install numpy opencv-python Pillow
    ```

4.  **Download Model Weights:**
    * The model weights (`starry_night_epoch_1.pth`) are **not included** due to their size.
    * Place the downloaded `.pth` file inside the **`saved_models/`** folder.

### Running the Real-Time Demo

Once everything is set up, run the script below to start the live, stylized webcam feed:

```bash
# Ensure you are inside the venv_style_transfer environment
python3 src/webcam_inference.py
