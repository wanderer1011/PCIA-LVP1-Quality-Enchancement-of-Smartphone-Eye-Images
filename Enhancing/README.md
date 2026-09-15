# Restormer Eye Image Enhancement

**Goal**: Transform low-quality smartphone eye images → slit-lamp clinical quality using a transformer-based model trained on paired data generated via Neural Style Transfer.

## Architecture: Restormer

[Restormer](https://arxiv.org/abs/2111.09881) (CVPR 2022) is the state-of-the-art for image restoration. It uses:

- **Multi-Dconv Head Transposed Attention (MDTA)**: Self-attention over channels (not spatial), making it O(C²) instead of O(N²) — efficient for 512×512 images
- **Gated-Dconv Feed-Forward Network (GDFN)**: Controlled info flow with depth-wise convolutions
- **4-level U-shaped encoder-decoder**: Multi-scale processing with skip connections
- **Residual learning**: Model predicts the enhancement residual (output = model(input) + input)

### Why Restormer over other options?
| Model | Pros | Cons | Verdict |
|-------|------|------|---------|
| **Restormer** | SOTA restoration quality, efficient transposed attention, proven on medical images | Needs `einops` package | **Best choice** |
| SwinIR | Good quality, Swin attention | Slower on 512×512, needs more VRAM | Runner-up |
| Pix2Pix | Simple GAN approach | Unstable training, mode collapse risk | Not recommended |
| U-Net alone | Fast, simple | No attention = misses global context | Too simple |

## Data Setup

Your paired data:
- **Input (degraded)**: `nst_results/` — NST-processed smartphone-style images (~1719 images)
- **Target (clean)**: `segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images/` — original slit-lamp quality
- **Pairing**: Matched by filename (1.png ↔ 1.png, 2.png ↔ 2.png, etc.)

## Installation

```bash
# From the PCIA root directory (where .venv is)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install einops tensorboard pillow
```

> If you already have PyTorch with CUDA, just: `pip install einops tensorboard`

## Quick Start

### 1. Verify the model loads
```bash
cd PCIA-LVP1-Quality-Enchancement-of-Smartphone-Eye-Images/Enhancing
python restormer_model.py
```
Expected output:
```
Restormer-Small: ~10M parameters
Input: torch.Size([1, 3, 256, 256]) -> Output: torch.Size([1, 3, 256, 256])
```

### 2. Train

**Recommended first run** (small model, default settings):
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

**On Windows (single-line)**:
```powershell
python train.py --input_dir ../nst_results --target_dir "../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images" --output_dir ./checkpoints --model_size small --epochs 200 --batch_size 4 --patch_size 256 --lr 3e-4
```

**If you have ≥12GB VRAM**, use the base model for better quality:
```powershell
python train.py --input_dir ../nst_results --target_dir "../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images" --output_dir ./checkpoints --model_size base --epochs 200 --batch_size 2 --patch_size 256 --lr 2e-4
```

**If running on CPU** (very slow, not recommended):
```powershell
python train.py --input_dir ../nst_results --target_dir "../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images" --output_dir ./checkpoints --model_size small --epochs 50 --batch_size 2 --patch_size 128 --no_amp
```

### 3. Monitor Training
```bash
tensorboard --logdir ./checkpoints/logs
```
Open http://localhost:6006 to see loss curves, PSNR, SSIM metrics.

### 4. Inference

**Enhance a single image:**
```powershell
python inference.py --checkpoint ./checkpoints/best_model.pth --model_size small --image "path/to/smartphone_eye.png" --output_dir ./enhanced_results
```

**Enhance an entire directory:**
```powershell
python inference.py --checkpoint ./checkpoints/best_model.pth --model_size small --input_dir ../nst_results --output_dir ./enhanced_results
```

**Create side-by-side comparisons (Input | Enhanced | Ground Truth):**
```powershell
python inference.py --checkpoint ./checkpoints/best_model.pth --model_size small --compare --input_dir ../nst_results --target_dir "../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images" --output_dir ./comparisons --num_compare 20
```

**For large images that don't fit in VRAM, use tiled inference:**
```powershell
python inference.py --checkpoint ./checkpoints/best_model.pth --model_size small --input_dir ./my_images --output_dir ./enhanced --tile_size 512
```

## Training Tips

### VRAM Guide
| Model | Patch Size | Batch Size | Approx. VRAM |
|-------|-----------|------------|---------------|
| small | 256×256 | 4 | ~6 GB |
| small | 256×256 | 8 | ~10 GB |
| base | 256×256 | 2 | ~8 GB |
| base | 256×256 | 4 | ~14 GB |

### If you run out of VRAM:
1. Reduce `--batch_size` (try 2 or 1)
2. Reduce `--patch_size` (try 128)
3. Use `--model_size small`
4. Use `--no_amp` (slower but sometimes uses less peak memory)

### Resume training from a checkpoint:
```powershell
python train.py --resume ./checkpoints/latest.pth --epochs 300 [other args...]
```

### Expected training performance:
- ~1700 pairs, small model, batch=4, 256×256 patches
- **GPU (RTX 3060+)**: ~2-3 min/epoch → ~7-10 hours for 200 epochs
- **Google Colab (T4)**: ~3-5 min/epoch → ~10-17 hours for 200 epochs
- **CPU**: ~20-30 min/epoch → not practical for full training

### Expected quality progression:
| Epoch | PSNR (approx) | Visual Quality |
|-------|---------------|----------------|
| 10 | ~22 dB | Colors improving, still blurry |
| 50 | ~26 dB | Good structure, some artifacts |
| 100 | ~28 dB | Sharp details, clean lighting |
| 200 | ~30+ dB | Near slit-lamp quality |

## Loss Function Breakdown

The training uses three complementary losses:

- **L1 Loss** (weight=1.0): Pixel-level fidelity — preserves anatomy
- **Perceptual/VGG Loss** (weight=0.1): Feature-level — natural textures and lighting
- **SSIM Loss** (weight=0.5): Structural similarity — preserves eye anatomy structure

## File Structure
```
Enhancing/
├── restormer_model.py   # Restormer architecture (MDTA + GDFN + U-Net decoder)
├── dataset.py           # Paired data loading with augmentation
├── losses.py            # L1 + VGG Perceptual + SSIM combined loss
├── train.py             # Training script with AMP, checkpointing, TensorBoard
├── inference.py         # Single image, batch, tiled, and comparison inference
└── README.md            # This file
```

## Output Files (after training)
```
checkpoints/
├── best_model.pth           # Best validation PSNR model
├── latest.pth               # Most recent checkpoint
├── checkpoint_epoch_25.pth  # Periodic checkpoints
├── samples/
│   ├── epoch_10.png         # Visual progress samples
│   └── epoch_20.png
├── logs/                    # TensorBoard logs
└── splits/
    ├── train.txt            # Training filenames
    └── val.txt              # Validation filenames
```
