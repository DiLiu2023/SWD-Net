#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test and compare all five models

Usage:
    python test_all_models.py --data_dir test_data --checkpoint_dir checkpoints

Author: Assistant
Date: 2024
"""

import subprocess
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def test_model(model_name, checkpoint_path, args):
    """Test a single model"""
    print("\n" + "="*80)
    print(f"Testing {model_name.upper()}")
    print("="*80 + "\n")
    
    output_dir = Path(args.output_dir) / model_name
    
    cmd = [
        sys.executable, "test.py",
        "--model", model_name,
        "--checkpoint", str(checkpoint_path),
        "--data_dir", args.data_dir,
        "--batch_size", str(args.batch_size),
        "--threshold", str(args.threshold),
        "--output_dir", str(output_dir),
        "--device", args.device
    ]
    
    if args.visualize:
        cmd.append("--visualize")
        cmd.extend(["--num_vis_samples", str(args.num_vis_samples)])
    
    try:
        start_time = datetime.now()
        subprocess.run(cmd, check=True)
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Load results
        results_path = output_dir / 'test_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print(f"\n✓ {model_name} testing completed in {duration}")
            return True, duration, results['metrics']['avg']
        else:
            print(f"\n✗ {model_name} results file not found")
            return False, duration, None
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} testing failed: {e}")
        return False, None, None


def find_checkpoint(checkpoint_dir, model_name):
    """Find best checkpoint for a model"""
    checkpoint_path = Path(checkpoint_dir)
    
    # Search patterns
    patterns = [
        f"{model_name}_*/best_model.pth",
        f"{model_name}/best_model.pth",
        f"best_{model_name}.pth"
    ]
    
    for pattern in patterns:
        matches = list(checkpoint_path.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def plot_comparison(all_results, save_path):
    """Plot comparison of all models"""
    models = list(all_results.keys())
    metrics = ['iou', 'f1', 'precision', 'recall', 'accuracy']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [all_results[model][metric] for model in models]
        
        bars = ax.bar(range(len(models)), values, color='skyblue', edgecolor='navy')
        
        # Highlight best model
        best_idx = np.argmax(values)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.upper()}')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nComparison plot saved to {save_path}")


def create_comparison_table(all_results, save_path):
    """Create comparison table"""
    models = list(all_results.keys())
    metrics = ['iou', 'f1', 'precision', 'recall', 'accuracy', 'specificity']
    
    # Create markdown table
    table = "# Model Performance Comparison\n\n"
    table += "| Model | IoU | F1 | Precision | Recall | Accuracy | Specificity |\n"
    table += "|-------|-----|----|-----------| -------|----------|-------------|\n"
    
    for model in models:
        row = f"| {model} |"
        for metric in metrics:
            value = all_results[model].get(metric, 0.0)
            row += f" {value:.4f} |"
        table += row + "\n"
    
    # Find best model for each metric
    table += "\n## Best Models per Metric\n\n"
    for metric in metrics:
        values = {model: all_results[model].get(metric, 0.0) for model in models}
        best_model = max(values, key=values.get)
        best_value = values[best_model]
        table += f"- **{metric.upper()}**: {best_model} ({best_value:.4f})\n"
    
    with open(save_path, 'w') as f:
        f.write(table)
    
    print(f"Comparison table saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test all SWD-Net models')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    # Test parameters
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='Number of visualization samples')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='test_results_all', help='Output directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+',
                        default=['unet', 'mamba_unet', 'swin_unet', 'segformer_unet', 'vit_unet'],
                        choices=['unet', 'mamba_unet', 'swin_unet', 'segformer_unet', 'vit_unet'],
                        help='Models to test')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SWD-Net: Testing All Models")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Test data: {args.data_dir}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Device: {args.device}")
    print()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test all models
    results = {}
    all_metrics = {}
    total_start = datetime.now()
    
    for model_name in args.models:
        # Find checkpoint
        checkpoint_path = find_checkpoint(args.checkpoint_dir, model_name)
        
        if checkpoint_path is None:
            print(f"\n✗ Checkpoint not found for {model_name}")
            results[model_name] = {'success': False, 'duration': None}
            continue
        
        print(f"Found checkpoint: {checkpoint_path}")
        
        # Test model
        success, duration, metrics = test_model(model_name, checkpoint_path, args)
        results[model_name] = {
            'success': success,
            'duration': duration,
            'checkpoint': str(checkpoint_path)
        }
        
        if success and metrics:
            all_metrics[model_name] = metrics
    
    total_end = datetime.now()
    total_duration = total_end - total_start
    
    # Print summary
    print("\n" + "="*80)
    print("TESTING SUMMARY")
    print("="*80)
    
    for model_name, result in results.items():
        if result['success']:
            print(f"✓ {model_name:20s} - Completed in {result['duration']}")
        else:
            print(f"✗ {model_name:20s} - Failed or no checkpoint")
    
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nTotal: {successful}/{len(args.models)} models tested successfully")
    print(f"Total time: {total_duration}")
    
    # Create comparison
    if all_metrics:
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        # Print table
        metrics_list = ['iou', 'f1', 'precision', 'recall', 'accuracy']
        print(f"\n{'Model':<20} {'IoU':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}")
        print("-" * 80)
        for model_name, metrics in all_metrics.items():
            print(f"{model_name:<20} ", end="")
            for metric in metrics_list:
                print(f"{metrics.get(metric, 0):<10.4f} ", end="")
            print()
        
        # Find best models
        print("\nBest models per metric:")
        for metric in metrics_list:
            values = {m: metrics.get(metric, 0) for m, metrics in all_metrics.items()}
            best = max(values, key=values.get)
            print(f"  {metric.upper():<15}: {best} ({values[best]:.4f})")
        
        # Create plots and tables
        plot_comparison(all_metrics, output_path / 'model_comparison.png')
        create_comparison_table(all_metrics, output_path / 'comparison_table.md')
        
        # Save all results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'models': results,
            'metrics': all_metrics,
            'total_duration': str(total_duration)
        }
        
        with open(output_path / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nAll results saved to {output_path}")
    
    print("="*80)


if __name__ == '__main__':
    main()




