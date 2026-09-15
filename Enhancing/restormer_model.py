"""
Restormer: Efficient Transformer for High-Resolution Image Restoration
Adapted for smartphone-to-slit-lamp eye image enhancement.

Architecture:
  - Multi-scale U-shaped encoder-decoder
  - Multi-Dconv Head Transposed Attention (MDTA)  
  - Gated-Dconv Feed-Forward Network (GDFN)
  - Skip connections at each level

Reference: Zamir et al., "Restormer: Efficient Transformer for High-Resolution
           Image Restoration", CVPR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention.
    
    Computes self-attention across channels (transposed) instead of spatial
    dimensions, making it O(C²) instead of O(N²) — efficient for high-res.
    Depth-wise convolutions encode local spatial context into each head.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network.
    
    Uses gating mechanism with depth-wise convolutions for controlled
    information flow, better than standard FFN for image restoration.
    """

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
            padding=1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2  # gating
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """Single Restormer transformer block: LayerNorm -> MDTA -> LayerNorm -> GDFN."""

    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        mu = x.mean(1, keepdim=True)
        sigma = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ---------------------------------------------------------------------------
# Downsampling / Upsampling
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Pixel-unshuffle based downsampling (2x)."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Pixel-shuffle based upsampling (2x)."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# Restormer
# ---------------------------------------------------------------------------

class Restormer(nn.Module):
    """
    Restormer for eye image enhancement (smartphone -> slit-lamp quality).

    4-level U-shaped encoder-decoder with transformer blocks at each level.
    Progressive channel expansion: dim -> 2*dim -> 4*dim -> 8*dim.

    Args:
        inp_channels:  Input image channels (3 for RGB).
        out_channels:  Output image channels (3 for RGB).
        dim:           Base feature dimension (default 48).
        num_blocks:    List of transformer blocks per level [enc1, enc2, enc3, bottleneck, dec3, dec2, dec1].
        num_heads:     List of attention heads per level.
        ffn_expansion_factor: FFN hidden dim multiplier.
        bias:          Use bias in conv layers.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8, 6, 6, 4],
        num_heads=[1, 2, 4, 8, 4, 2, 1],
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # --- Encoder ---
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, num_heads[0], ffn_expansion_factor, bias)
              for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[1], ffn_expansion_factor, bias)
              for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, num_heads[2], ffn_expansion_factor, bias)
              for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(dim * 8, num_heads[3], ffn_expansion_factor, bias)
              for _ in range(num_blocks[3])]
        )

        # --- Decoder ---
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, num_heads[4], ffn_expansion_factor, bias)
              for _ in range(num_blocks[4])]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[5], ffn_expansion_factor, bias)
              for _ in range(num_blocks[5])]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, num_heads[6], ffn_expansion_factor, bias)
              for _ in range(num_blocks[6])]
        )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        """
        Args:
            inp_img: (B, 3, H, W) degraded smartphone eye image.
        Returns:
            (B, 3, H, W) enhanced image (residual learning: output + input).
        """
        inp_enc_level1 = self.patch_embed(inp_img)

        # Encoder
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)

        # Bottleneck
        latent = self.bottleneck(inp_enc_level4)

        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Residual connection: learn the enhancement residual
        out = self.output(out_dec_level1) + inp_img

        return out


# ---------------------------------------------------------------------------
# Lightweight variant for faster training / lower VRAM
# ---------------------------------------------------------------------------

def restormer_small(inp_channels=3, out_channels=3):
    """Smaller Restormer: ~10M params, good for ~1700 image pairs."""
    return Restormer(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=32,
        num_blocks=[2, 3, 3, 4, 3, 3, 2],
        num_heads=[1, 2, 4, 8, 4, 2, 1],
        ffn_expansion_factor=2.66,
        bias=False,
    )


def restormer_base(inp_channels=3, out_channels=3):
    """Base Restormer: ~26M params, better quality if you have the VRAM."""
    return Restormer(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=48,
        num_blocks=[4, 6, 6, 8, 6, 6, 4],
        num_heads=[1, 2, 4, 8, 4, 2, 1],
        ffn_expansion_factor=2.66,
        bias=False,
    )


if __name__ == "__main__":
    # Quick test
    model = restormer_small()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Restormer-Small: {total_params:.2f}M parameters")

    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    model = restormer_base()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Restormer-Base: {total_params:.2f}M parameters")
