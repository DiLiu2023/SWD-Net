#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train all five models sequentially

Usage:
    python train_all_models.py --data_dir ../water_planets/default --epochs 100

Author: Assistant
Date: 2024
"""

import subprocess
import argparse
import sys
from datetime import datetime


def train_model(model_name, args):
    """Train a single model"""
    print("\n" + "="*80)
    print(f"Training {model_name.upper()}")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable, "train.py",
        "--model", model_name,
        "--data_dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patch_size", str(args.patch_size),
        "--overlap", str(args.overlap),
        "--val_split", str(args.val_split),
        "--loss", args.loss,
        "--save_dir", args.save_dir,
        "--save_freq", str(args.save_freq),
        "--device", args.device
    ]
    
    try:
        start_time = datetime.now()
        subprocess.run(cmd, check=True)
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✓ {model_name} training completed in {duration}")
        return True, duration
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} training failed: {e}")
        return False, None


def main():
    parser = argparse.ArgumentParser(description='Train all SWD-Net models')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--overlap', type=int, default=128, help='Patch overlap')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='combined', 
                        choices=['combined', 'focal', 'dice', 'tversky'],
                        help='Loss function')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='Device to use')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['unet', 'mamba_unet', 'swin_unet', 'segformer_unet', 'vit_unet'],
                        choices=['unet', 'mamba_unet', 'swin_unet', 'segformer_unet', 'vit_unet'],
                        help='Models to train')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SWD-Net: Training All Models")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print()
    
    # Train all models
    results = {}
    total_start = datetime.now()
    
    for model_name in args.models:
        success, duration = train_model(model_name, args)
        results[model_name] = {'success': success, 'duration': duration}
    
    total_end = datetime.now()
    total_duration = total_end - total_start
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for model_name, result in results.items():
        if result['success']:
            print(f"✓ {model_name:20s} - Completed in {result['duration']}")
        else:
            print(f"✗ {model_name:20s} - Failed")
    
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nTotal: {successful}/{len(args.models)} models trained successfully")
    print(f"Total time: {total_duration}")
    print("="*80)


if __name__ == '__main__':
    main()




