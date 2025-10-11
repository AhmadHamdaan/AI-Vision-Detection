#!/usr/bin/env python3
"""
Simple YOLOv12 Inference Script
"""

import argparse
from ultralytics import YOLO
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv12 inference')
    parser.add_argument('--weights', required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--source', required=True, help='Image path, folder, or URL')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', default='0', help='GPU device')
    parser.add_argument('--save', action='store_true', help='Save results')
    
    args = parser.parse_args()
    
    if not Path(args.weights).exists():
        print(f"❌ Model not found: {args.weights}")
        return
    
    print(f"🔍 Running inference...")
    print(f"🤖 Model: {args.weights}")
    print(f"📸 Source: {args.source}")
    print(f"🎯 Confidence: {args.conf}")
    
    try:
        # Load model
        model = YOLO(args.weights)
        
        # Run inference
        results = model.predict(
            source=args.source,
            conf=args.conf,
            device=args.device,
            save=args.save,
            show=True  # Show results in window
        )
        
        print("✅ Inference completed!")
        
        # Print detection summary
        for r in results:
            if r.boxes is not None:
                print(f"📊 Detected {len(r.boxes)} objects")
                for box in r.boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)
                    print(f"  - Class {cls}: {conf:.2f} confidence")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

if __name__ == "__main__":
    main()
