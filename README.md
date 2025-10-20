# YOLOv12 Simple Setup



## Quick Start

### 1. Install Dependencies
```bash
# Install PyTorch with CUDA 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r simple_requirements.txt
```

### 2. Run Setup Script
```bash
python setup_simple.py
```

### 3. Train with COCO128 (Ready to Use!)
```bash
# Quick test (5 epochs) - Recommended to start
python simple_train.py --dataset coco128 --epochs 5 --batch 8

# Full training (50 epochs)
python simple_train.py --dataset coco128 --epochs 50 --batch 16

# Different model sizes
python simple_train.py --dataset coco128 --model yolov12s.pt --epochs 50 --batch 12
python simple_train.py --dataset coco128 --model yolov12m.pt --epochs 50 --batch 8
```

### 4. Add Your Own Dataset (Optional)
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

### 5. Run Inference
```bash
# Test with COCO128 trained model
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source your_image.jpg

# Test with pretrained model (no training needed)
python simple_infer.py --weights yolov12n.pt --source your_image.jpg

# Test with your own trained model
python simple_infer.py --weights runs/train_your_dataset/weights/best.pt --source your_image.jpg
```

## Optimized Settings for RTX 3070 Ti

| Model | Batch Size | Image Size | VRAM Usage | Training Speed |
|-------|------------|------------|------------|----------------|
| yolov12n | 32 | 640 | ~4GB | Very Fast |
| yolov12s | 16 | 640 | ~6GB | Fast |
| yolov12m | 8 | 640 | ~8GB | Medium |

## File Structure
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

## Commands

### Training Commands
```bash
# COCO128 Training (Ready to use!)
python simple_train.py --dataset coco128 --epochs 5 --batch 8
python simple_train.py --dataset coco128 --epochs 50 --batch 16
python simple_train.py --dataset coco128 --model yolov12s.pt --epochs 50 --batch 12

# Custom Dataset Training
python simple_train.py --dataset my_dataset --epochs 50 --batch 16
python simple_train.py --dataset my_dataset --model yolov12s.pt --epochs 100 --batch 16 --imgsz 832
```

### Inference Commands
```bash
# COCO128 trained model
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source image.jpg

# Pretrained model (no training needed)
python simple_infer.py --weights yolov12n.pt --source image.jpg

# Custom trained model
python simple_infer.py --weights runs/train_my_dataset/weights/best.pt --source image.jpg

# Batch inference (folder)
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source images_folder/

# With confidence threshold
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source image.jpg --conf 0.5
```

## GPU Monitoring

Check GPU usage during training:
```bash
# Windows
nvidia-smi

# Or watch continuously
watch -n 1 nvidia-smi
```

## Expected Performance

### Training Times 
- **yolov12n**: ~2 min/epoch (1000 images)
- **yolov12s**: ~3 min/epoch (1000 images)  
- **yolov12m**: ~5 min/epoch (1000 images)

### Inference Speed
- **yolov12n**: ~150 FPS
- **yolov12s**: ~120 FPS
- **yolov12m**: ~80 FPS

## Troubleshooting

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

## Command Reference

### Start Training (Recommended)
```bash
# Quick test first (5 epochs) - Start here!
python simple_train.py --dataset coco128 --epochs 5 --batch 8

# Full training (50 epochs)
python simple_train.py --dataset coco128 --epochs 50 --batch 16
```

### Different Model Sizes
```bash
# Nano (fastest, smallest)
python simple_train.py --dataset coco128 --model yolov12n.pt --epochs 50 --batch 16

# Small (balanced)
python simple_train.py --dataset coco128 --model yolov12s.pt --epochs 50 --batch 12

# Medium (most accurate)
python simple_train.py --dataset coco128 --model yolov12m.pt --epochs 50 --batch 8
```

### Run Inference
```bash
# SHOW ONLY (doesn't save - just displays results)
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source your_image.jpg

# SAVE RESULTS (recommended - saves processed image)
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source your_image.jpg --save

# Saved results go to: yolov12/runs/detect/predict/
```

### Auto-Save Examples
```bash
# Single image with save
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source "C:\Users\Ahmad Hamdaan\Desktop\test.jpg" --save

# Batch folder with save (processes all images in folder)
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source images_folder/ --save

# With custom confidence and save
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source your_image.jpg --conf 0.5 --save

# Test with pretrained model (no training needed)
python simple_infer.py --weights yolov12n.pt --source your_image.jpg --save
```

# Batch inference on folder
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source images_folder/

# With custom confidence
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source your_image.jpg --conf 0.5
```

### Where to Put Test Images
You can put test images **anywhere** and use the full path:
- **Desktop**: `C:\Users\Ahmad Hamdaan\Desktop\test.jpg`
- **Downloads**: `C:\Users\Ahmad Hamdaan\Downloads\my_image.jpg`
- **Project folder**: `test_image.jpg` (if in same folder as scripts)
- **Create a folder**: `mkdir test_images` then put images there

## Training Progress

Your training results will be saved in:
- **Model weights**: `runs/train_coco128/weights/best.pt`
- **Training plots**: `runs/train_coco128/`
- **Results**: `runs/train_coco128/results.png`

## Performance Tips

- **Start with 5 epochs** to test everything works
- **Use batch size 8-16** for RTX 3070 Ti (8GB VRAM)
- **Monitor GPU usage** with `nvidia-smi`
- **yolov12n** is fastest for testing
- **yolov12s** gives best balance of speed/accuracy

## Troubleshooting

### Path with Spaces Error
```bash
# Wrong (causes error)
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source C:\Users\Ahmad Hamdaan\Pictures\Camera Roll\test.jpg

# Correct (use quotes)
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source "C:\Users\Ahmad Hamdaan\Pictures\Camera Roll\test.jpg"
```

### File Not Found Error
```bash
# Check if file exists first
dir "C:\Users\Ahmad Hamdaan\Pictures\Camera Roll\sidetest.jpg"

# Or use a simpler path (copy image to desktop)
copy "C:\Users\Ahmad Hamdaan\Pictures\Camera Roll\sidetest.jpg" "C:\Users\Ahmad Hamdaan\Desktop\sidetest.jpg"
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source "C:\Users\Ahmad Hamdaan\Desktop\sidetest.jpg" --save
```

### Multiprocessing Errors
- The training script is already fixed with `workers=0`
- If issues persist, restart your terminal

### CUDA Out of Memory
- Reduce batch size: `--batch 4` or `--batch 8`
- Use smaller model: `--model yolov12n.pt`

### Quick Fix for Path Issues
```bash
# Copy image to project folder (easiest)
copy "C:\Users\Ahmad Hamdaan\Pictures\Camera Roll\sidetest.jpg" "sidetest.jpg"
python simple_infer.py --weights runs/train_coco128/weights/best.pt --source sidetest.jpg --save
```

## File Locations

- **Dataset**: `datasets/coco128/` (128 images, 80 classes)
- **Config**: `datasets/coco128.yaml`
- **Trained models**: `runs/train_coco128/weights/`
- **Training logs**: `runs/train_coco128/`

## That's It!

This simple setup gives you:
- YOLOv12 training on your RTX 3070 Ti
- COCO128 dataset ready to use
- Easy dataset management
- Fast inference
- No complex Docker setup
- Perfect for personal projects

**Start with the quick test command above to verify everything works!**
