
#!/usr/bin/env python3
"""
Simple YOLOv12 Training Script
Optimized for RTX 3070 Ti (8GB VRAM)
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import yaml

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv12 model')
    parser.add_argument('--dataset', required=True, help='Dataset name (folder in datasets/)')
    parser.add_argument('--model', default='yolov12n.pt', choices=['yolov12n.pt', 'yolov12s.pt', 'yolov12m.pt'], 
                       help='Model size (nano=fast, small=balanced, medium=accurate)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (16 for RTX 3070 Ti)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='0', help='GPU device (0 for RTX 3070 Ti)')
    
    args = parser.parse_args()
    
    # Dataset path
    dataset_path = Path(f"datasets/{args.dataset}")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Available datasets:")
        for d in Path("datasets").glob("*"):
            if d.is_dir():
                print(f"  - {d.name}")
        return
    
    # Check if COCO128 (uses different structure)
    if args.dataset == 'coco128':
        # COCO128 already has its own yaml file in the datasets directory
        data_yaml_path = f"datasets/{args.dataset}.yaml"
        if not Path(data_yaml_path).exists():
            print(f"❌ COCO128 yaml file not found: {data_yaml_path}")
            return
    else:
        # Create data.yaml for custom datasets
        data_yaml = {
            'train': str(dataset_path / 'images' / 'train'),
            'val': str(dataset_path / 'images' / 'val'),
            'nc': 80,  # Default COCO classes, will be auto-detected
            'names': []  # Will be auto-detected
        }
        
        # Save temporary data.yaml
        data_yaml_path = f"temp_data_{args.dataset}.yaml"
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
    
    print(f"🚀 Starting YOLOv12 training...")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🤖 Model: {args.model}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch: {args.batch}")
    print(f"🖼️  Image size: {args.imgsz}")
    print(f"🎮 GPU: {args.device}")
    
    try:
        # Load model
        model = YOLO(args.model)
        
        # Train
        results = model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project='runs',
            name=f'train_{args.dataset}',
            exist_ok=True,
            half=True,  # Mixed precision for RTX 3070 Ti
            augment=True,  # Data augmentation
            verbose=True,
            workers=0  # Fix multiprocessing issues on Windows
        )
        
        print("✅ Training completed!")
        print(f"📁 Results saved in: runs/train_{args.dataset}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        # Cleanup temporary files only
        temp_yaml_path = f"temp_data_{args.dataset}.yaml"
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

if __name__ == "__main__":
    main()
