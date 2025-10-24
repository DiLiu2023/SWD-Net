#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing/Evaluation script for SWD-Net models

Usage:
    python test.py --model unet --checkpoint checkpoints/best_model.pth --data_dir test_data

Author: Assistant
Date: 2024
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from models import create_model
from dataset import RemoteSensingDataset
from torch.utils.data import DataLoader


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate comprehensive metrics"""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate confusion matrix components
    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))
    
    epsilon = 1e-7
    
    # Calculate metrics
    iou = tp / (tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    
    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def test_model(model, test_loader, device, threshold=0.5):
    """Test model on test dataset"""
    model.eval()
    
    all_metrics = {
        'iou': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'specificity': []
    }
    
    all_predictions = []
    all_targets = []
    
    print("Testing model...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate metrics for this batch
            metrics = calculate_metrics(outputs, masks, threshold)
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # Store predictions for confusion matrix
            pred = torch.sigmoid(outputs)
            pred_binary = (pred > threshold).float()
            all_predictions.extend(pred_binary.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {key: np.std(values) for key, values in all_metrics.items()}
    
    return avg_metrics, std_metrics, all_predictions, all_targets


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Water'],
                yticklabels=['Background', 'Water'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_metrics_comparison(metrics, save_path):
    """Plot metrics bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics comparison saved to {save_path}")


def visualize_predictions(model, test_dataset, device, save_dir, num_samples=10, threshold=0.5):
    """Visualize predictions on sample images"""
    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Denormalization parameters
    norm_mean = np.array([0.3514572472166658, 0.39416361084365814, 0.320239625615125])
    norm_std = np.array([0.13550397467320616, 0.10947113210718354, 0.09616026742437417])
    
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Visualizing predictions"):
        image, mask = test_dataset[idx]
        
        # Predict
        with torch.no_grad():
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred > threshold).astype(np.float32)
        
        # Denormalize image
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = image_np * norm_std + norm_mean
        image_np = np.clip(image_np, 0, 1)
        
        # Get mask
        mask_np = mask.squeeze().cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Prediction (Probability)')
        axes[2].axis('off')
        
        axes[3].imshow(pred_binary, cmap='gray')
        axes[3].set_title(f'Prediction (Binary, th={threshold})')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / f'sample_{idx}.png', dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test SWD-Net models')
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet', 'mamba_unet', 'swin_unet', 'segformer_unet', 'vit_unet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--n_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=1, help='Number of output classes')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset directory')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--overlap', type=int, default=128, help='Patch overlap')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    # Test parameters
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    parser.add_argument('--visualize', action='store_true', help='Create visualization samples')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='Number of visualization samples')
    
    # Save parameters
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Load model
    print(f"\nLoading {args.model} model...")
    model = create_model(args.model, n_channels=args.n_channels, n_classes=args.n_classes)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Model loaded successfully")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = RemoteSensingDataset(
        images_dir=os.path.join(args.data_dir, 'images'),
        masks_dir=os.path.join(args.data_dir, 'masks'),
        augment=False,
        patch_size=args.patch_size,
        overlap=args.overlap
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test model
    print("\n" + "="*60)
    avg_metrics, std_metrics, all_predictions, all_targets = test_model(
        model, test_loader, device, args.threshold
    )
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for metric, value in avg_metrics.items():
        print(f"{metric.upper():15s}: {value:.4f} ± {std_metrics[metric]:.4f}")
    print("="*60)
    
    # Save results
    results = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'threshold': args.threshold,
        'metrics': {
            'avg': {k: float(v) for k, v in avg_metrics.items()},
            'std': {k: float(v) for k, v in std_metrics.items()}
        }
    }
    
    results_path = output_path / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    # Sample predictions for confusion matrix (to avoid memory issues with large datasets)
    sample_size = min(100000, len(all_predictions))
    sample_indices = np.random.choice(len(all_predictions), sample_size, replace=False)
    sampled_preds = [all_predictions[i] for i in sample_indices]
    sampled_targets = [all_targets[i] for i in sample_indices]
    
    plot_confusion_matrix(
        sampled_targets,
        sampled_preds,
        output_path / 'confusion_matrix.png'
    )
    
    # Plot metrics
    print("Generating metrics plot...")
    plot_metrics_comparison(
        avg_metrics,
        output_path / 'metrics_comparison.png'
    )
    
    # Visualize predictions
    if args.visualize:
        print("\nGenerating visualization samples...")
        visualize_predictions(
            model,
            test_dataset,
            device,
            output_path / 'visualizations',
            num_samples=args.num_vis_samples,
            threshold=args.threshold
        )
    
    print("\n" + "="*60)
    print("Testing completed!")
    print(f"All results saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()



