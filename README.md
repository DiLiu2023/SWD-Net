# SWD-Net: A Benchmark Dataset for Small Water Body Detection from High-Resolution Aerial Imagery

A comprehensive deep learning framework for small and seasonal water body segmentation in remote sensing imagery, implementing five baseline architectures for comparative analysis.


## 📋 Overview

This project implements five state-of-the-art deep learning architectures for semantic segmentation of water bodies:

1. **U-Net** - Original U-Net architecture [Ronneberger et al., 2015]
2. **Mamba-UNet** - U-Net with Mamba state space blocks for efficient sequence modeling
3. **Swin-UNet** - U-Net with Swin Transformer hierarchical attention blocks
4. **SegFormer-based U-Net** - Custom architecture combining SegFormer's efficient attention encoder [Xie et al., 2021] with U-Net decoder
5. **ViT-based U-Net** - Custom architecture combining Vision Transformer encoder [Dosovitskiy et al., 2021] with U-Net decoder

## 🗂️ Project Structure

```
SWD-Net/
├── models.py           # Model architectures
├── dataset.py          # Dataset loading and preprocessing
├── losses.py           # Loss functions
├── train.py            # Training script
├── predict.py          # Prediction script
├── test.py             # Testing/evaluation script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
<img width="501" height="664" alt="image" src="https://github.com/user-attachments/assets/c706ed91-b935-4ad0-b798-b20d8e8a7607" />

Fig. 1. Representative SWD-Net samples. Top: 2023 NAIP Summer Images; bottom: 2024 Google Earth Winter Images.

## 🔧 Installation
### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset Structure

Download datasets and weights  as follows:
https://drive.google.com/drive/folders/1jly3jbQV86yJIFRmOhpgWsDPbzyFi5xv?usp=drive_link
```
SWD-Net/
├── train/
│   ├── images/          # RGB images
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── masks/           # Binary masks
│       ├── image1.png
│       ├── image2.png
│       └── ...
```
## 🚀 Usage

### 1. Training

Train a model using the following command:

```bash
# Train U-Net
python train.py --model unet --data_dir ../SWD-Net/train --epochs 100 --batch_size 8 --lr 1e-4

# Train Mamba-UNet
python train.py --model mamba_unet --data_dir ../SWD-Net/train --epochs 100 --batch_size 8 --lr 1e-4

# Train Swin-UNet
python train.py --model swin_unet --data_dir ../SWD-Net/train --epochs 100 --batch_size 8 --lr 1e-4

# Train SegFormer-based U-Net
python train.py --model segformer_unet --data_dir ../SWD-Net/train --epochs 100 --batch_size 8 --lr 1e-4

# Train ViT-based U-Net
python train.py --model vit_unet --data_dir ../SWD-Net/train --epochs 100 --batch_size 8 --lr 1e-4
```

#### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | Required |
| `--data_dir` | Path to dataset | Required |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size | 8 |
| `--lr` | Learning rate | 1e-4 |
| `--patch_size` | Patch size for training | 256 |
| `--overlap` | Patch overlap | 128 |
| `--val_split` | Validation split ratio | 0.2 |
| `--loss` | Loss function (combined/focal/dice/tversky) | combined |
| `--save_dir` | Directory to save checkpoints | checkpoints |
| `--save_freq` | Save checkpoint frequency | 10 |
| `--device` | Device (cuda/cpu) | cuda |

### 2. Prediction

Predict on new images:

```bash
# Predict on a single image
python predict.py --model unet --checkpoint weights/SWIN-UNET_20250729_161647_best.pth \
    --input test_images/image.png --output predictions

# Predict on a directory of images
python predict.py --model unet --checkpoint weights/SWIN-UNET_20250729_161647_best.pth \
    --input test_images/ --output predictions --save_probability
```

#### Prediction Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | Required |
| `--checkpoint` | Path to checkpoint | Required |
| `--input` | Input image or directory | Required |
| `--output` | Output directory | Required |
| `--patch_size` | Patch size for sliding window | 256 |
| `--stride` | Stride for sliding window | 128 |
| `--threshold` | Binary threshold | 0.5 |
| `--save_probability` | Save probability maps | False |
| `--device` | Device (cuda/cpu) | cuda |

### 3. Testing/Evaluation

Evaluate model performance on test set:

```bash
python test.py --model unet --checkpoint weights/SWIN-UNET_20250729_161647_best.pth \
    --data_dir test_data --visualize --num_vis_samples 10
```

#### Testing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | Required |
| `--checkpoint` | Path to checkpoint | Required |
| `--data_dir` | Path to test dataset | Required |
| `--threshold` | Binary threshold | 0.5 |
| `--batch_size` | Batch size | 8 |
| `--visualize` | Create visualizations | False |
| `--num_vis_samples` | Number of visualization samples | 10 |
| `--output_dir` | Output directory | test_results |
| `--device` | Device (cuda/cpu) | cuda |

## 📈 Evaluation Metrics

The framework computes the following metrics:

- **IoU (Intersection over Union)**: Overlap between prediction and ground truth
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positive predictions
- **Recall**: Ratio of detected positive samples
- **Accuracy**: Overall pixel-wise accuracy
- **Specificity**: True negative rate

## 🏗️ Model Architectures

### 1. U-Net
Original encoder-decoder architecture with skip connections.
- Parameters: ~31M
- Suitable for: General segmentation tasks

### 2. Mamba-UNet
U-Net enhanced with Mamba state space model blocks for efficient sequence modeling.
- Parameters: ~35M
- Suitable for: Long-range dependency modeling

### 3. Swin-UNet
U-Net with hierarchical Swin Transformer blocks and shifted window attention.
- Parameters: ~41M
- Suitable for: Multi-scale feature extraction

### 4. SegFormer-based U-Net
Custom architecture with SegFormer's efficient attention mechanism as encoder.
- Parameters: ~38M
- Suitable for: Efficient attention-based segmentation

### 5. ViT-based U-Net
Custom architecture with Vision Transformer encoder for global context modeling.
- Parameters: ~42M
- Suitable for: Global feature extraction

## 🎯 Loss Functions

The framework supports multiple loss functions:

- **Combined Loss**: Weighted combination of Dice + Focal + Tversky (default)
- **Focal Loss**: Addresses class imbalance
- **Dice Loss**: Optimizes overlap directly
- **Tversky Loss**: Balances false positives and false negatives

## 📝 Citation

If you use SWD-Net in your research, please cite:

```bibtex
@dataset{swd_net_2025,
  title={SWD-Net: Surface Water Detection Network},
  author={Di Liu， Chengbin Deng},
  year={2025},
  url={https://github.com/yourusername/SWD-Net}
}
```

### Model References

1. U-Net: Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
2. SegFormer: Xie, E., et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS 2021.
3. ViT: Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- U-Net implementation based on PyTorch
- Swin Transformer concepts from Microsoft Research
- SegFormer architecture from NVIDIA
- Vision Transformer from Google Research













