"""
Combined loss functions for eye image enhancement training.

Uses a weighted combination of:
  - L1 (pixel-level fidelity)
  - Perceptual / VGG loss (feature-level similarity)
  - SSIM loss (structural similarity)

These three together ensure the enhanced images are:
  1. Pixel-accurate (L1)
  2. Perceptually natural (VGG features)
  3. Structurally consistent with anatomy (SSIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-19 features (layers relu1_2, relu2_2, relu3_4, relu4_4)."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Extract features at specific layers
        self.slice1 = nn.Sequential(*vgg[:4])    # relu1_2
        self.slice2 = nn.Sequential(*vgg[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*vgg[9:18])  # relu3_4
        self.slice4 = nn.Sequential(*vgg[18:27]) # relu4_4

        # Freeze
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        x, y = pred, target

        for slc in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = slc(x)
            with torch.no_grad():
                y = slc(y)
            loss += F.l1_loss(x, y)

        return loss


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: 1 - SSIM."""

    def __init__(self, window_size=11, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer('window', self._create_window(window_size, channels))

    @staticmethod
    def _gaussian(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, channels):
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        pad = self.window_size // 2

        mu1 = F.conv2d(pred, self.window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of L1 + Perceptual + SSIM losses.
    
    Default weights tuned for medical image enhancement:
      - L1 (w=1.0): Strong pixel fidelity for anatomy preservation
      - Perceptual (w=0.1): Natural-looking textures and lighting
      - SSIM (w=0.5): Structural consistency
    """

    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, ssim_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        total = (self.l1_weight * l1 +
                 self.perceptual_weight * perceptual +
                 self.ssim_weight * ssim)

        return total, {
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'ssim': ssim.item(),
            'total': total.item(),
        }
