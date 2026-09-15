"""
Training script for Restormer eye image enhancement.

Trains the model to enhance NST-degraded smartphone-style eye images
to slit-lamp quality using paired data.

Features:
  - Progressive training (small patches -> larger patches)
  - Cosine annealing LR with warm restarts
  - Mixed precision (AMP) for faster training
  - Checkpointing with best-model tracking
  - TensorBoard logging
  - Validation with PSNR / SSIM metrics
"""

import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image, make_grid

from restormer_model import restormer_small, restormer_base
from dataset import create_train_val_split
from losses import CombinedLoss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse.item()))


def calculate_ssim_metric(pred, target):
    """Structural Similarity Index (simplified, for logging)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = pred.mean()
    mu2 = target.mean()
    sigma1_sq = pred.var()
    sigma2_sq = target.var()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = {'l1': 0.0, 'perceptual': 0.0, 'ssim': 0.0, 'total': 0.0}
    running_psnr = 0.0
    n_batches = 0

    for batch_idx, (inp, tgt, _) in enumerate(loader):
        inp, tgt = inp.to(device), tgt.to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=scaler.is_enabled()):
            pred = model(inp)
            pred = pred.clamp(0, 1)
            loss, loss_dict = criterion(pred, tgt)

        scaler.scale(loss).backward()
        # Gradient clipping to stabilize training
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for k in running_loss:
            running_loss[k] += loss_dict[k]
        running_psnr += calculate_psnr(pred.detach(), tgt)
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss['total'] / n_batches
            avg_psnr = running_psnr / n_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

    for k in running_loss:
        running_loss[k] /= n_batches
    running_psnr /= n_batches
    return running_loss, running_psnr


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = {'l1': 0.0, 'perceptual': 0.0, 'ssim': 0.0, 'total': 0.0}
    running_psnr = 0.0
    running_ssim = 0.0
    n_batches = 0

    for inp, tgt, _ in loader:
        inp, tgt = inp.to(device), tgt.to(device)

        with autocast('cuda', enabled=True):
            pred = model(inp)
            pred = pred.clamp(0, 1)
            loss, loss_dict = criterion(pred, tgt)

        for k in running_loss:
            running_loss[k] += loss_dict[k]
        running_psnr += calculate_psnr(pred, tgt)
        running_ssim += calculate_ssim_metric(pred, tgt)
        n_batches += 1

    for k in running_loss:
        running_loss[k] /= n_batches
    return running_loss, running_psnr / n_batches, running_ssim / n_batches


def save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_psnr, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_psnr': best_psnr,
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('best_psnr', 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Restormer for eye image enhancement")

    # Data
    parser.add_argument('--input_dir', type=str,
                        default='../nst_results',
                        help='Directory with degraded (smartphone-style) images')
    parser.add_argument('--target_dir', type=str,
                        default='../segementation/Data Generation/Data_Raw/Original_Slit-lamp_Images',
                        help='Directory with target slit-lamp images')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Where to save checkpoints and logs')

    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base'],
                        help='small (~10M params) or base (~26M params)')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Random crop size for training (256 recommended)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--perceptual_weight', type=float, default=0.1)
    parser.add_argument('--ssim_weight', type=float, default=0.5)

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

    # Data
    train_ds, val_ds = create_train_val_split(
        args.input_dir, args.target_dir,
        val_ratio=args.val_ratio,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    if args.model_size == 'small':
        model = restormer_small()
    else:
        model = restormer_base()

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: Restormer-{args.model_size} ({total_params:.2f}M params)")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(args.l1_weight, args.perceptual_weight, args.ssim_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 4), T_mult=1, eta_min=args.min_lr
    )
    scaler = GradScaler('cuda', enabled=not args.no_amp)

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    start_epoch = 0
    best_psnr = 0.0

    # Resume
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, best_psnr = load_checkpoint(
            args.resume, model, optimizer, scaler, scheduler, device
        )
        print(f"  Resuming from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # Training
    print(f"\n{'='*60}")
    print(f"Training Restormer-{args.model_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Learning rate: {args.lr} -> {args.min_lr}")
    print(f"  Loss: L1({args.l1_weight}) + Perceptual({args.perceptual_weight}) + SSIM({args.ssim_weight})")
    print(f"  AMP: {'disabled' if args.no_amp else 'enabled'}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        scheduler.step()

        # Validate
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.0f}s) | LR: {lr:.2e}")
        print(f"  Train - Loss: {train_loss['total']:.4f} | PSNR: {train_psnr:.2f} dB")
        print(f"  Val   - Loss: {val_loss['total']:.4f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")

        writer.add_scalars('Loss', {'train': train_loss['total'], 'val': val_loss['total']}, epoch)
        writer.add_scalars('PSNR', {'train': train_psnr, 'val': val_psnr}, epoch)
        writer.add_scalar('Val/SSIM', val_ssim, epoch)
        writer.add_scalar('LR', lr, epoch)

        for k in ['l1', 'perceptual', 'ssim']:
            writer.add_scalars(f'Loss/{k}', {'train': train_loss[k], 'val': val_loss[k]}, epoch)

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_psnr,
                            os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  >> New best PSNR: {best_psnr:.2f} dB (saved)")

        # Save periodic checkpoint
        if epoch % 25 == 0:
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_psnr,
                            os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Save sample images every 10 epochs (and always on the last/only epoch)
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                sample_inp, sample_tgt, _ = next(iter(val_loader))
                sample_inp = sample_inp.to(device)
                sample_tgt = sample_tgt.to(device)
                with autocast('cuda', enabled=not args.no_amp):
                    sample_pred = model(sample_inp).clamp(0, 1)

                grid = make_grid(
                    torch.cat([sample_inp, sample_pred, sample_tgt], dim=0),
                    nrow=sample_inp.shape[0], normalize=False,
                )
                save_image(grid, os.path.join(args.output_dir, 'samples', f'epoch_{epoch}.png'))

        # Save latest
        save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_psnr,
                        os.path.join(args.output_dir, 'latest.pth'))

    writer.close()
    print(f"\nTraining complete! Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
