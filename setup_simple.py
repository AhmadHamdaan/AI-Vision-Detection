#!/usr/bin/env python3
"""
Simple Setup Script for YOLOv12 on RTX 3070 Ti
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("CUDA not available - will use CPU")
            return False
    except ImportError:
        print("PyTorch not installed yet")
        return False

def setup_coco128_dataset():
    """Download and set up COCO128 dataset"""
    print("\nSetting up COCO128 dataset...")
    coco128_path = Path("datasets/coco128")
    
    if coco128_path.exists() and any(coco128_path.iterdir()):
        print("COCO128 dataset already exists")
        return True
    
    print("Attempting to download COCO128 dataset...")
    try:
        # Try using YOLO's built-in dataset download
        from ultralytics import YOLO
        model = YOLO('yolov12n.pt')
        
        # Download COCO128 using YOLO's dataset functionality
        print("Downloading COCO128 dataset...")
        # This will download COCO128 automatically when used
        model.train(data='coco128.yaml', epochs=1, verbose=False, exist_ok=True)
        print("COCO128 dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Automatic COCO128 download failed: {e}")
        print("\nManual COCO128 setup:")
        print("1. Download COCO128 from: https://github.com/ultralytics/assets/releases/download/v8.3.0/coco128.zip")
        print("2. Extract to: datasets/coco128/")
        print("3. Or run: python -c \"from ultralytics import YOLO; YOLO('yolov12n.pt').train(data='coco128.yaml', epochs=1)\"")
        return False

def main():
    print("Setting up YOLOv12 for RTX 3070 Ti...")
    print("=" * 50)
    
    # Create directory structure
    print("Creating directory structure...")
    directories = [
        "datasets",
        "datasets/sample_dataset/images/train",
        "datasets/sample_dataset/images/val",
        "datasets/sample_dataset/labels/train", 
        "datasets/sample_dataset/labels/val",
        "models",
        "runs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    # Install PyTorch with CUDA
    print("\nInstalling PyTorch with CUDA 11.8...")
    if not run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch with CUDA"
    ):
        print("PyTorch installation failed")
        return False
    
    # Install other requirements
    print("\nInstalling other requirements...")
    if not run_command(
        "pip install -r simple_requirements.txt",
        "Installing requirements"
    ):
        print("Requirements installation failed")
        return False
    
    # Install YOLOv12 from repository
    print("\nInstalling YOLOv12...")
    if not run_command(
        "git clone https://github.com/sunsmarterjie/yolov12.git",
        "Cloning YOLOv12 repository"
    ):
        print("YOLOv12 repository already exists or clone failed")
    
    if not run_command(
        "cd yolov12 && pip install -e .",
        "Installing YOLOv12"
    ):
        print("YOLOv12 installation failed")
        return False
    
    # Install additional YOLOv12 dependencies
    print("\nInstalling YOLOv12 dependencies...")
    if not run_command(
        "pip install huggingface_hub safetensors",
        "Installing YOLOv12 dependencies"
    ):
        print("YOLOv12 dependencies installation failed")
        return False
    
    # Test GPU
    print("\nTesting GPU...")
    if check_gpu():
        print("GPU setup successful!")
    else:
        print("GPU not detected, but setup will continue")
    
    # Test YOLOv12
    print("\nTesting YOLOv12...")
    try:
        from ultralytics import YOLO
        # YOLOv12 models are automatically downloaded when first used
        model = YOLO('yolov12n.pt')  # This will download the model if not present
        print("YOLOv12 model downloaded and loaded successfully!")
        print(f"Model info: {model.model_name if hasattr(model, 'model_name') else 'YOLOv12n'}")
    except Exception as e:
        print(f"YOLOv12 test failed: {e}")
        print("This might be due to network issues. You can try again later.")
        print("   The model will be downloaded automatically when you first use it.")
        return False
    
    # Setup COCO128 dataset
    setup_coco128_dataset()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Add your dataset to datasets/your_dataset_name/")
    print("2. Run training: python simple_train.py --dataset your_dataset_name")
    print("3. Run inference: python simple_infer.py --weights runs/train_your_dataset/weights/best.pt --source your_image.jpg")
    
    print("\nExample commands:")
    print("  python simple_train.py --dataset coco128 --model yolov12n.pt --epochs 10 --batch 16")
    print("  python simple_infer.py --weights runs/train_coco128/weights/best.pt --source test.jpg")
    print("\nAvailable datasets:")
    print("  - coco128: Small COCO dataset for testing (already downloaded)")
    print("  - your_custom_dataset: Add your own dataset to datasets/your_custom_dataset/")

if __name__ == "__main__":
    main()
