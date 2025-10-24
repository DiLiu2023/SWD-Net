#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions for SWD-Net training

Author: Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss for handling imbalanced data"""
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combined loss function with optimal weights"""
    
    def __init__(self, dice_weight=0.2, focal_weight=0.4, tversky_weight=0.4):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        self.dice = DiceLoss(smooth=1.0)
        self.focal = FocalLoss(alpha=0.25, gamma=3.0)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        
        return (self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss + 
                self.tversky_weight * tversky_loss)


def get_loss_function(loss_type='combined', **kwargs):
    """
    Factory function to get loss function
    
    Args:
        loss_type: One of ['combined', 'focal', 'dice', 'tversky']
        **kwargs: Additional loss-specific parameters
    
    Returns:
        loss_fn: The requested loss function
    """
    losses = {
        'combined': CombinedLoss,
        'focal': FocalLoss,
        'dice': DiceLoss,
        'tversky': TverskyLoss
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(losses.keys())}")
    
    return losses[loss_type](**kwargs)


if __name__ == '__main__':
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    pred = torch.randn(4, 1, 256, 256).to(device)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float().to(device)
    
    # Test all losses
    for loss_name in ['focal', 'dice', 'tversky', 'combined']:
        print(f"\nTesting {loss_name} loss...")
        loss_fn = get_loss_function(loss_name)
        loss_fn = loss_fn.to(device)
        loss = loss_fn(pred, target)
        print(f"{loss_name} loss: {loss.item():.4f}")



