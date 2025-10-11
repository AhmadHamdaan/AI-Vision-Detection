# YOLOv12 Simple Setup - RTX 3070 Ti

A streamlined YOLOv12 setup optimized for your RTX 3070 Ti (8GB VRAM).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install PyTorch with CUDA (for RTX 3070 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r simple_requirements.txt
```

### 2. Run Setup Script
```bash
python setup_simple.py
```

### 3. Add Your Dataset
```
datasets/
└── your_dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   └── val/
    │       ├── image3.jpg
    │       └── image4.jpg
    └── labels/ (for YOLO format)
        ├── train/
        │   ├── image1.txt
        │   └── image2.txt
        └── val/
            ├── image3.txt
            └── image4.txt
```

### 4. Train Your Model
```bash
# Train with YOLOv12 nano (fastest)
python simple_train.py --dataset your_dataset --model yolov12n.pt --epochs 50 --batch 16

# Train with YOLOv12 small (better accuracy)
python simple_train.py --dataset your_dataset --model yolov12s.pt --epochs 50 --batch 16
```

### 5. Run Inference
```bash
python simple_infer.py --weights runs/train_your_dataset/weights/best.pt --source your_image.jpg
```

## 🎯 Optimized Settings for RTX 3070 Ti

| Model | Batch Size | Image Size | VRAM Usage | Training Speed |
|-------|------------|------------|------------|----------------|
| yolov12n | 32 | 640 | ~4GB | Very Fast |
| yolov12s | 16 | 640 | ~6GB | Fast |
| yolov12m | 8 | 640 | ~8GB | Medium |

## 📁 File Structure
```
AI-Vision-Detection/
├── simple_train.py          # Training script
├── simple_infer.py          # Inference script
├── setup_simple.py          # Setup script
├── simple_requirements.txt  # Dependencies
├── datasets/                # Your datasets go here
├── models/                  # Trained models
└── runs/                    # Training outputs
```

## 🔧 Commands

### Training
```bash
# Basic training
python simple_train.py --dataset my_dataset

# Advanced training
python simple_train.py --dataset my_dataset --model yolov12s.pt --epochs 100 --batch 16 --imgsz 832
```

### Inference
```bash
# Single image
python simple_infer.py --weights runs/train_my_dataset/weights/best.pt --source image.jpg

# Batch inference (folder)
python simple_infer.py --weights runs/train_my_dataset/weights/best.pt --source images_folder/

# With confidence threshold
python simple_infer.py --weights runs/train_my_dataset/weights/best.pt --source image.jpg --conf 0.5
```

## 🎮 GPU Monitoring

Check GPU usage during training:
```bash
# Windows
nvidia-smi

# Or watch continuously
watch -n 1 nvidia-smi
```

## 📊 Expected Performance

### Training Times (RTX 3070 Ti)
- **yolov12n**: ~2 min/epoch (1000 images)
- **yolov12s**: ~3 min/epoch (1000 images)  
- **yolov12m**: ~5 min/epoch (1000 images)

### Inference Speed
- **yolov12n**: ~150 FPS
- **yolov12s**: ~120 FPS
- **yolov12m**: ~80 FPS

## 🛠️ Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python simple_train.py --dataset my_dataset --batch 8

# Use smaller model
python simple_train.py --dataset my_dataset --model yolov12n.pt
```

### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎉 That's It!

This simple setup gives you:
- ✅ YOLOv12 training on your RTX 3070 Ti
- ✅ Easy dataset management
- ✅ Fast inference
- ✅ No complex Docker setup
- ✅ Perfect for personal projects

Start with `yolov12n.pt` for quick experiments, then move to `yolov12s.pt` or `yolov12m.pt` for better accuracy!
