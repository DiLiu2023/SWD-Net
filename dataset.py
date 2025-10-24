#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset and utilities for SWD-Net training

Author: Assistant
Date: 2024
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF


class RemoteSensingDataset(Dataset):
    """Remote sensing dataset with patch extraction and augmentation"""
    
    def __init__(self, images_dir, masks_dir, augment=False, image_size=1024, patch_size=256, overlap=128):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.image_size = image_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Normalization statistics (computed from training data)
        self.norm_mean = [0.3514572472166658, 0.39416361084365814, 0.320239625615125]
        self.norm_std = [0.13550397467320616, 0.10947113210718354, 0.09616026742437417]
        
        # Calculate number of patches per image
        self.patches_per_image = []
        for img_name in self.image_files:
            img_path = os.path.join(images_dir, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                h, w = image.shape[:2]
                stride = patch_size - overlap
                num_patches_h = max(1, (h - patch_size) // stride + 1)
                num_patches_w = max(1, (w - patch_size) // stride + 1)
                self.patches_per_image.append(num_patches_h * num_patches_w)
            else:
                self.patches_per_image.append(0)
        
        self.total_patches = sum(self.patches_per_image)
        
    def __len__(self):
        return self.total_patches

    def _get_image_and_patch_indices(self, idx):
        """Get image index and patch index from global index"""
        current_sum = 0
        for i, num_patches in enumerate(self.patches_per_image):
            if idx < current_sum + num_patches:
                image_idx = i
                patch_idx = idx - current_sum
                return image_idx, patch_idx
            current_sum += num_patches
        return 0, 0

    def _extract_patch(self, image, mask, patch_idx):
        """Extract patch from image"""
        h, w = image.shape[:2]
        stride = self.patch_size - self.overlap
        
        num_patches_w = max(1, (w - self.patch_size) // stride + 1)
        row_idx = patch_idx // num_patches_w
        col_idx = patch_idx % num_patches_w
        
        start_y = min(row_idx * stride, h - self.patch_size)
        start_x = min(col_idx * stride, w - self.patch_size)
        
        image_patch = image[start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]
        mask_patch = mask[start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]
        
        return image_patch, mask_patch

    def __getitem__(self, idx):
        # Get image and patch indices
        image_idx, patch_idx = self._get_image_and_patch_indices(idx)
        img_name = self.image_files[image_idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Try different mask naming conventions
        mask_paths = [
            os.path.join(self.masks_dir, f"{os.path.splitext(img_name)[0]}.png"),
            os.path.join(self.masks_dir, f"{os.path.splitext(img_name)[0]}_mask.png"),
            os.path.join(self.masks_dir, img_name)
        ]
        mask_path = None
        for path in mask_paths:
            if os.path.exists(path):
                mask_path = path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"Mask file not found: {img_name}")
        
        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        
        # Extract patch
        image_patch, mask_patch = self._extract_patch(image, mask, patch_idx)
        
        # Convert to PIL
        image_patch = Image.fromarray(image_patch)
        mask_patch = Image.fromarray(mask_patch)
        
        if self.augment:
            # Geometric transformations (synchronized)
            if torch.rand(1) < 0.5:
                image_patch = TF.hflip(image_patch)
                mask_patch = TF.hflip(mask_patch)

            if torch.rand(1) < 0.5:
                image_patch = TF.vflip(image_patch)
                mask_patch = TF.vflip(mask_patch)
            
            # Random rotation
            if torch.rand(1) < 0.3:
                angle = transforms.RandomRotation.get_params([-30, 30])
                image_patch = TF.rotate(image_patch, angle)
                mask_patch = TF.rotate(mask_patch, angle)

            # Random resized crop
            if torch.rand(1) < 0.4:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image_patch, scale=(0.85, 1.0), ratio=(0.9, 1.1)
                )
                image_patch = TF.resized_crop(image_patch, i, j, h, w, size=(self.patch_size, self.patch_size))
                mask_patch = TF.resized_crop(mask_patch, i, j, h, w, size=(self.patch_size, self.patch_size))

            # Color jitter (image only)
            if torch.rand(1) < 0.6:
                image_patch = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15)(image_patch)
            
            # Gaussian blur (image only)
            if torch.rand(1) < 0.3:
                image_patch = transforms.GaussianBlur(kernel_size=3)(image_patch)
            
            # Add noise (image only)
            if torch.rand(1) < 0.2:
                image_patch = transforms.ToTensor()(image_patch)
                noise = torch.randn_like(image_patch) * 0.05
                image_patch = torch.clamp(image_patch + noise, 0, 1)
                image_patch = transforms.ToPILImage()(image_patch)
        
        # Convert to tensor
        image_patch = TF.to_tensor(image_patch)
        
        # Binarize mask
        mask_np = np.array(mask_patch)
        mask_binary = (mask_np > 0).astype(np.float32)
        mask_patch = torch.from_numpy(mask_binary).unsqueeze(0)
        
        # Normalize image
        image_patch = TF.normalize(image_patch, mean=self.norm_mean, std=self.norm_std)

        return image_patch, mask_patch


if __name__ == '__main__':
    # Test dataset
    import matplotlib.pyplot as plt
    
    dataset = RemoteSensingDataset(
        images_dir='path/to/images',
        masks_dir='path/to/masks',
        augment=True,
        patch_size=256,
        overlap=128
    )
    
    print(f"Total patches: {len(dataset)}")
    
    # Visualize some samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        img, mask = dataset[i]
        
        # Denormalize image
        img = img * torch.tensor(dataset.norm_std).view(3, 1, 1) + torch.tensor(dataset.norm_mean).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title(f'Image {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mask {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    print("Saved dataset_samples.png")



