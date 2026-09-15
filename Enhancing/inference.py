"""
Inference script for the trained Restormer enhancement model.

Enhances smartphone eye images to slit-lamp quality.
Supports single image, batch directory, and tiled inference for large images.
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from restormer_model import restormer_small, restormer_base


def load_model(checkpoint_path, model_size='small', device='cuda'):
    """Load trained Restormer from checkpoint."""
    if model_size == 'small':
        model = restormer_small()
    else:
        model = restormer_base()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get('epoch', '?')
    psnr = ckpt.get('best_psnr', '?')
    print(f"Loaded model from epoch {epoch} (best PSNR: {psnr})")
    return model


@torch.no_grad()
def enhance_image(model, image_path, device='cuda', tile_size=None, tile_overlap=32):
    """Enhance a single image.
    
    Args:
        model: Trained Restormer model.
        image_path: Path to input image.
        device: Device to run on.
        tile_size: If set, process in tiles (for images that don't fit in VRAM).
        tile_overlap: Overlap between tiles to avoid boundary artifacts.
    
    Returns:
        Enhanced image tensor (3, H, W) in [0, 1].
    """
    img = Image.open(image_path).convert("RGB")
    inp = transforms.ToTensor()(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    if tile_size and (inp.shape[2] > tile_size or inp.shape[3] > tile_size):
        result = tiled_inference(model, inp, tile_size, tile_overlap, device)
    else:
        with autocast(enabled=(device != 'cpu')):
            result = model(inp).clamp(0, 1)

    return result.squeeze(0).cpu()


@torch.no_grad()
def tiled_inference(model, inp, tile_size, overlap, device):
    """Process large images in overlapping tiles to save VRAM."""
    _, _, h, w = inp.shape
    output = torch.zeros_like(inp)
    weight = torch.zeros_like(inp)

    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = inp[:, :, y_start:y_end, x_start:x_end]

            with autocast(enabled=True):
                pred = model(tile).clamp(0, 1)

            output[:, :, y_start:y_end, x_start:x_end] += pred
            weight[:, :, y_start:y_end, x_start:x_end] += 1.0

    return output / weight


def enhance_directory(model, input_dir, output_dir, device='cuda', tile_size=None):
    """Enhance all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])

    print(f"Found {len(images)} images in {input_dir}")
    total_time = 0

    for i, img_path in enumerate(images):
        t0 = time.time()
        enhanced = enhance_image(model, img_path, device, tile_size)
        elapsed = time.time() - t0
        total_time += elapsed

        out_path = output_dir / img_path.name
        save_image(enhanced, out_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            avg_time = total_time / (i + 1)
            remaining = avg_time * (len(images) - i - 1)
            print(f"  [{i+1}/{len(images)}] {img_path.name} ({elapsed:.1f}s) "
                  f"| ETA: {remaining/60:.1f} min")

    print(f"\nDone! {len(images)} images enhanced in {total_time:.0f}s "
          f"({total_time/len(images):.1f}s/image)")
    print(f"Results saved to: {output_dir}")


def create_comparison(model, input_dir, target_dir, output_dir, device='cuda', n=10):
    """Create side-by-side comparison: Input | Enhanced | Ground Truth."""
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = {f.name for f in input_dir.glob("*.png")}
    target_files = {f.name for f in target_dir.glob("*.png")}
    paired = sorted(input_files & target_files)[:n]

    to_tensor = transforms.ToTensor()

    for fname in paired:
        inp_img = to_tensor(Image.open(input_dir / fname).convert("RGB"))
        tgt_img = to_tensor(Image.open(target_dir / fname).convert("RGB"))

        enhanced = enhance_image(model, input_dir / fname, device)

        # Ensure same size
        min_h = min(inp_img.shape[1], tgt_img.shape[1], enhanced.shape[1])
        min_w = min(inp_img.shape[2], tgt_img.shape[2], enhanced.shape[2])
        inp_img = inp_img[:, :min_h, :min_w]
        enhanced = enhanced[:, :min_h, :min_w]
        tgt_img = tgt_img[:, :min_h, :min_w]

        comparison = torch.cat([inp_img, enhanced, tgt_img], dim=2)  # side by side
        save_image(comparison, output_dir / f"compare_{fname}")

    print(f"Saved {len(paired)} comparisons to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhance eye images using trained Restormer")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base'])

    # Input modes (choose one)
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image to enhance')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory of images to enhance')
    parser.add_argument('--output_dir', type=str, default='./enhanced_results',
                        help='Where to save enhanced images')

    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Create input/enhanced/target comparisons')
    parser.add_argument('--target_dir', type=str, default=None,
                        help='Target directory for comparisons')
    parser.add_argument('--num_compare', type=int, default=20,
                        help='Number of comparison images to generate')

    # Processing
    parser.add_argument('--tile_size', type=int, default=None,
                        help='Tile size for large images (e.g., 512). None = process whole image')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')

    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.model_size, device)

    if args.compare and args.input_dir and args.target_dir:
        create_comparison(model, args.input_dir, args.target_dir,
                          args.output_dir, device, args.num_compare)
    elif args.image:
        enhanced = enhance_image(model, args.image, device, args.tile_size)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, Path(args.image).name)
        save_image(enhanced, out_path)
        print(f"Enhanced image saved to: {out_path}")
    elif args.input_dir:
        enhance_directory(model, args.input_dir, args.output_dir, device, args.tile_size)
    else:
        parser.print_help()
        print("\nProvide --image for a single image or --input_dir for batch processing.")


if __name__ == '__main__':
    main()
