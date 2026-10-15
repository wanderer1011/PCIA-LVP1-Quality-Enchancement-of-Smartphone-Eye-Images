# PCIA-LVP1: Quality Enhancement of Smartphone Eye Images

A deep learning pipeline for transforming low-quality smartphone-captured eye images into slit-lamp clinical quality using state-of-the-art transformer-based image restoration and semantic segmentation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
  - [1. Data Generation & Segmentation](#1-data-generation--segmentation)
  - [2. Restormer Training](#2-restormer-training)
  - [3. Testing & Inference](#3-testing--inference)
- [Model Details](#model-details)
- [Results](#results)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

This project addresses a critical challenge in teleophthalmology: **smartphone-captured eye images are often of insufficient quality for reliable clinical diagnosis**. The pipeline uses a two-stage approach:

1. **Data Generation & Segmentation**: A lightweight U-Net++ model performs semantic segmentation of eye anatomy (conjunctiva, cornea, pupil, lesions) from annotated slit-lamp images. A degrader script then creates paired training data by simulating smartphone-quality degradation.

2. **Image Enhancement**: A Restormer (Restoration Transformer) model — the state-of-the-art in high-resolution image restoration — is trained on paired data (degraded smartphone-style images → clean slit-lamp images) generated via Neural Style Transfer (NST). The goal is to transform low-quality smartphone eye images into images of clinical slit-lamp quality.

---

## Key Features

- **Transformer-Based Restoration**: Uses Restormer (CVPR 2022) for efficient high-resolution image restoration.
- **Lightweight Segmentation**: U-Net++ with EfficientNet-B0 encoder for fast, accurate eye anatomy segmentation.
- **Automated Data Generation**: Degrader script creates paired degraded/clean datasets from annotated slit-lamp images.
- **Multi-Class Segmentation**: 5 classes — Background, Conjunctiva, Cornea, Pupil, and Lesion.
- **Neural Style Transfer Pipeline**: Generates realistic smartphone-style degraded images for training paired data.
- **Production-Ready**: Includes training, testing, and inference scripts with configurable model sizes.

---

## Project Structure

```
PCIA-LVP1-Quality-Enchancement-of-Smartphone-Eye-Images/
├── Enhancing/                          # Image enhancement module
│   ├── restormer_model.py              # Restormer architecture definition
│   ├── train.py                        # Training script
│   ├── test.py                         # Inference script
│   └── checkpoints/                    # Saved model weights
│
├── segementation/                      # Segmentation module
│   ├── Using_UNET.py                   # U-Net++ training script
│   ├── Data Generation/
│   │   ├── degrader.py                 # Paired data generation script
│   │   ├── Data_Raw/
│   │   │   ├── Original_Slit-lamp_Images/   # Clean slit-lamp images
│   │   │   └── Annotations.csv              # Region annotations
│   │   └── Data_Degraded/
│   │       ├── images/                 # Degraded images
│   │       └── masks/                  # Segmentation masks
│   └── Testing/
│       ├── test_Unet.py                # U-Net testing script
│       └── test_Unet_with_patching.py  # Patched inference
│
├── nst_results/                        # NST-processed smartphone images
├── .gitignore
└── README.md
```

---

## Architecture

### Restormer (Image Enhancement)

Restormer is a Transformer-based model designed for high-resolution image restoration. Key components:

- **Multi-Dconv Head Transposed Attention (MDTA)**: Self-attention over channels (not spatial), reducing complexity from O(N²) to O(C²) — efficient for 512×512 images.
- **Gated-Dconv Feed-Forward Network (GDFN)**: Controlled information flow with depth-wise convolutions.
- **4-Level U-Shaped Encoder-Decoder**: Multi-scale processing with skip connections.
- **Residual Learning**: Model predicts the enhancement residual (output = model(input) + input).

**Why Restormer?** A comparison of alternatives:

| Model | Pros | Cons | Verdict |
|-------|------|------|---------|
| **Restormer** | SOTA restoration quality, efficient transposed attention, proven on medical images | Needs `einops` package | **Best choice** |
| SwinIR | Good quality, Swin attention | Slower on 512×512, needs more VRAM | Runner-up |
| Pix2Pix | Simple GAN approach | Unstable training, mode collapse risk | Not recommended |
| U-Net alone | Fast, simple | No attention = misses global context | Too simple |

### U-Net++ (Segmentation)

- **Encoder**: EfficientNet-B0 (pre-trained on ImageNet) — lightweight and fast.
- **Decoder**: U-Net++ with nested skip connections.
- **Classes**: 5 (Background, Conjunctiva, Cornea, Pupil, Lesion).
- **Input Size**: 512×512.
- **Loss**: CrossEntropyLoss with raw logits.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, ≥12GB VRAM for base model)
- PyTorch 2.0+

### Step 1: Clone the Repository

```bash
git clone https://github.com/wanderer1011/PCIA-LVP1-Quality-Enchancement-of-Smartphone-Eye-Images.git
cd PCIA-LVP1-Quality-Enchancement-of-Smartphone-Eye-Images
```

### Step 2: Install Dependencies

From the PCIA root directory (where `.venv` is):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install einops tensorboard pillow segmentation-models-pytorch opencv-python albumentations pandas tqdm
```

> **Note**: If you already have PyTorch with CUDA installed, just run: `pip install einops tensorboard`.

---

## Data Setup

The pipeline expects paired data in the following structure:

| Type | Path | Description |
|------|------|-------------|
| **Input (degraded)** | `nst_results/` | NST-processed smartphone-style images (~1719 images) |
| **Target (clean)** | `segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images/` | Original slit-lamp quality images |
| **Pairing** | — | Matched by filename (1.png ↔ 1.png, 2.png ↔ 2.png, etc.) |

### Data Generation Pipeline

1. **Annotation**: Annotate slit-lamp images with region labels (Background, Conjunctiva, Cornea, Pupil, Lesion) in `Annotations.csv`.
2. **Mask Generation**: Run `degrader.py` to generate segmentation masks from annotations.
3. **Degradation**: The degrader creates paired degraded images and masks using smart resize (preserving class IDs).
4. **NST Processing**: Apply Neural Style Transfer to degraded images to produce `nst_results/`.

---

## Usage

### 1. Data Generation & Segmentation

**Generate Segmentation Masks:**

```bash
cd segementation/Data\ Generation
python degrader.py
```

**Train U-Net++ Segmentation Model:**

```bash
cd segementation
python Using_UNET.py
```

This will:
- Load images and masks from `Data Generation/Data_Degraded/`
- Train for 10 epochs (configurable)
- Save the model as `eye_segmentation_unet.pth`

**Test Segmentation:**

```bash
cd segementation/Testing
python test_Unet.py
# or with patching
python test_Unet_with_patching.py
```

### 2. Restormer Training

**Verify the Model Loads:**

```bash
cd Enhancing
python restormer_model.py
```

Expected output:
```
Restormer-Small: ~10M parameters
Input: torch.Size([1, 3, 256, 256]) -> Output: torch.Size([1, 3, 256, 256])
```

**Train (Recommended First Run):**

```bash
python train.py \
    --input_dir ../nst_results \
    --target_dir "../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images" \
    --output_dir ./checkpoints \
    --model_size small \
    --epochs 200 \
    --batch_size 4 \
    --patch_size 256 \
    --lr 3e-4
```

**On Windows (single-line):**

```bash
python train.py --input_dir ../nst_results --target_dir "../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images" --output_dir ./checkpoints --model_size small --epochs 200 --batch_size 4 --patch_size 256 --lr 3e-4
```

**If you have ≥12GB VRAM**, use the base model for better quality:

```bash
python train.py --input_dir ../nst_results --target_dir "../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images" --output_dir ./checkpoints --model_size base --epochs 200 --batch_size 2 --patch_size 256 --lr 3e-4
```

### 3. Testing & Inference

```bash
cd Enhancing
python test.py --checkpoint ./checkpoints/best_model.pth --input_dir ../nst_results --output_dir ./results
```

---

## Model Details

### Restormer Configuration

| Parameter | Small Model | Base Model |
|-----------|-------------|------------|
| Parameters | ~10M | ~26M |
| VRAM Required | ~6GB | ≥12GB |
| Recommended Batch Size | 4 | 2 |
| Patch Size | 256×256 | 256×256 |
| Learning Rate | 3e-4 | 3e-4 |

### U-Net++ Configuration

| Parameter | Value |
|-----------|-------|
| Encoder | EfficientNet-B0 |
| Encoder Weights | ImageNet |
| Classes | 5 |
| Input Size | 512×512 |
| Batch Size | 8 |
| Learning Rate | 0.0001 |
| Epochs | 10 (configurable) |
| Loss | CrossEntropyLoss |

### Class Mapping

| Class ID | Label |
|----------|-------|
| 0 | Background |
| 1 | Conjunctiva |
| 2 | Cornea |
| 3 | Pupil |
| 4 | Lesion |
| 5 | Flash (specular highlights) |

---

## Results

*(Add your experimental results here, including PSNR, SSIM, and visual comparisons.)*

### Metrics

| Model | PSNR | SSIM | LPIPS |
|-------|------|------|-------|
| Baseline (no enhancement) | — | — | — |
| Restormer-Small | — | — | — |
| Restormer-Base | — | — | — |

---

## Contributors

- **Abdur Rehman** — Contributor
- **wanderer1011** — Project maintainer

---

## Acknowledgments

- **Restormer**: Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR 2022.
- **U-Net++**: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation."
- **segmentation_models_pytorch**: For providing pre-trained encoders.
- **Neural Style Transfer**: For generating paired training data.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{pcia_lvp1_2026,
  title={PCIA-LVP1: Quality Enhancement of Smartphone Eye Images},
  author={Abdur Rehman and wanderer1011},
  year={2026},
  url={https://github.com/wanderer1011/PCIA-LVP1-Quality-Enchancement-of-Smartphone-Eye-Images}
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch_size` or use `--model_size small` |
| `einops` not found | `pip install einops` |
| Mask values out of range | Ensure masks use `INTER_NEAREST` interpolation |
| Slow training | Use a smaller patch size or reduce epochs |

---

## Roadmap

- [ ] Add pre-trained model weights
- [ ] Implement ONNX/TensorRT export for deployment
- [ ] Add more segmentation classes (e.g., iris, sclera)
- [ ] Integrate real-time smartphone inference
- [ ] Publish results on public ophthalmology datasets

---

*Last updated: February 2026*
