#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWD-Net Models - Five baseline architectures for water segmentation

Models:
1. U-Net: Original U-Net architecture
2. Mamba-UNet: U-Net with Mamba state space blocks
3. Swin-UNet: U-Net with Swin Transformer blocks
4. SegFormer-based U-Net: SegFormer encoder with U-Net decoder
5. ViT-based U-Net: Vision Transformer encoder with U-Net decoder

Author: Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# ============================================================================
# Basic Building Blocks
# ============================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Mamba State Space Block
# ============================================================================

class MambaBlock(nn.Module):
    """Mamba state space model block"""
    def __init__(self, dim, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand_factor = expand_factor
        
        # SSM parameters - Fixed dimensions
        expanded_dim = dim * expand_factor
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_state, expanded_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(expanded_dim, d_state) * 0.1)
        self.D = nn.Parameter(torch.randn(expanded_dim) * 0.1)
        
        # Projection layers
        self.in_proj = nn.Sequential(
            nn.Linear(dim, expanded_dim),
            nn.Dropout(dropout)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(expanded_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # Reshape for sequence processing
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        
        # Project input
        x = self.in_proj(x)  # (B, H*W, expanded_dim)
        
        # SSM computation (simplified - process as a whole sequence)
        state = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(x.size(1)):
            x_t = x[:, t, :]  # (B, expanded_dim)
            # Update state: state = tanh(A @ state + B @ x_t)
            state = torch.tanh(torch.matmul(state, self.A.t()) + torch.matmul(x_t, self.B.t()))
            # Output: y_t = C @ state + D * x_t
            y_t = torch.matmul(state, self.C.t()) + x_t * self.D
            outputs.append(y_t)
        
        x = torch.stack(outputs, dim=1)
        x = self.out_proj(x)
        
        # Reshape back
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + residual


# ============================================================================
# Swin Transformer Block
# ============================================================================

def window_partition(x, window_size):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        shortcut = x
        
        x = self.norm1(x)
        
        # Pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = x.shape[1], x.shape[2]
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Attention
        attn_windows = self.attn(x_windows)
        
        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        return x


# ============================================================================
# Model 1: U-Net
# ============================================================================

class UNet(nn.Module):
    """Original U-Net architecture"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================================
# Model 2: Mamba-UNet
# ============================================================================

class MambaUNet(nn.Module):
    """U-Net with Mamba state space blocks"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, d_state=16):
        super(MambaUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.mamba1 = MambaBlock(64, d_state=d_state)
        
        self.down1 = Down(64, 128)
        self.mamba2 = MambaBlock(128, d_state=d_state)
        
        self.down2 = Down(128, 256)
        self.mamba3 = MambaBlock(256, d_state=d_state)
        
        self.down3 = Down(256, 512)
        self.mamba4 = MambaBlock(512, d_state=d_state)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.mamba5 = MambaBlock(1024 // factor, d_state=d_state)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.mamba1(x1)
        
        x2 = self.down1(x1)
        x2 = self.mamba2(x2)
        
        x3 = self.down2(x2)
        x3 = self.mamba3(x3)
        
        x4 = self.down3(x3)
        x4 = self.mamba4(x4)
        
        x5 = self.down4(x4)
        x5 = self.mamba5(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================================
# Model 3: Swin-UNet
# ============================================================================

class SwinUNet(nn.Module):
    """U-Net with Swin Transformer blocks"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, window_size=7, num_heads=4):
        super(SwinUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.swin1 = SwinTransformerBlock(64, num_heads, window_size, shift_size=0)
        
        self.down1 = Down(64, 128)
        self.swin2 = SwinTransformerBlock(128, num_heads, window_size, shift_size=window_size//2)
        
        self.down2 = Down(128, 256)
        self.swin3 = SwinTransformerBlock(256, num_heads, window_size, shift_size=0)
        
        self.down3 = Down(256, 512)
        self.swin4 = SwinTransformerBlock(512, num_heads, window_size, shift_size=window_size//2)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.swin5 = SwinTransformerBlock(1024 // factor, num_heads, window_size, shift_size=0)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.swin1(x1)
        
        x2 = self.down1(x1)
        x2 = self.swin2(x2)
        
        x3 = self.down2(x2)
        x3 = self.swin3(x3)
        
        x4 = self.down3(x3)
        x4 = self.swin4(x4)
        
        x5 = self.down4(x4)
        x5 = self.swin5(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================================
# Model 4: SegFormer-based U-Net
# ============================================================================

class EfficientAttention(nn.Module):
    """Efficient attention with spatial reduction"""
    def __init__(self, dim, num_heads=8, sr_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        q = self.q(x).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class SegFormerUNet(nn.Module):
    """SegFormer-based U-Net with efficient attention"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, sr_ratio=4, num_heads=8):
        super(SegFormerUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Overlapping patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.inc = DoubleConv(64, 64)
        self.eff_attn1 = EfficientAttention(64, num_heads, sr_ratio)
        
        self.down1 = Down(64, 128)
        self.eff_attn2 = EfficientAttention(128, num_heads, sr_ratio)
        
        self.down2 = Down(128, 256)
        self.eff_attn3 = EfficientAttention(256, num_heads, sr_ratio)
        
        self.down3 = Down(256, 512)
        self.eff_attn4 = EfficientAttention(512, num_heads, sr_ratio)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.eff_attn5 = EfficientAttention(1024 // factor, num_heads, sr_ratio)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        
        x1 = self.inc(x)
        x1 = self.eff_attn1(x1)
        
        x2 = self.down1(x1)
        x2 = self.eff_attn2(x2)
        
        x3 = self.down2(x2)
        x3 = self.eff_attn3(x3)
        
        x4 = self.down3(x3)
        x4 = self.eff_attn4(x4)
        
        x5 = self.down4(x4)
        x5 = self.eff_attn5(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================================
# Model 5: ViT-based U-Net
# ============================================================================

class ViTUNet(nn.Module):
    """Vision Transformer based U-Net"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, patch_size=16, embed_dim=384, depth=6, num_heads=6):
        super(ViTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Encoder with CNN blocks
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Transformer encoder on bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024 // factor,
            nhead=8,
            dim_feedforward=(1024 // factor) * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply transformer on bottleneck
        B, C, H, W = x5.shape
        x_flat = x5.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_flat = self.transformer(x_flat)
        x5 = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_type='unet', n_channels=3, n_classes=1, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: One of ['unet', 'mamba_unet', 'swin_unet', 'segformer_unet', 'vit_unet']
        n_channels: Number of input channels (default: 3)
        n_classes: Number of output classes (default: 1)
        **kwargs: Additional model-specific parameters
    
    Returns:
        model: The requested model
    """
    models = {
        'unet': UNet,
        'mamba_unet': MambaUNet,
        'swin_unet': SwinUNet,
        'segformer_unet': SegFormerUNet,
        'vit_unet': ViTUNet
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](n_channels=n_channels, n_classes=n_classes, **kwargs)


if __name__ == '__main__':
    # Test all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 256, 256).to(device)
    
    for model_name in ['unet', 'mamba_unet', 'swin_unet', 'segformer_unet', 'vit_unet']:
        print(f"\nTesting {model_name}...")
        model = create_model(model_name).to(device)
        with torch.no_grad():
            y = model(x)
        print(f"Input shape: {x.shape}, Output shape: {y.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

