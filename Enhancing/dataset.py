"""
Paired dataset for smartphone (NST-degraded) <-> slit-lamp eye image enhancement.

Loads matched pairs by filename from two directories:
  - input_dir:  NST-degraded images (nst_results/)
  - target_dir: Original slit-lamp images (Original_Slit-lamp_Images/)

Supports random cropping, flipping, and rotation for training augmentation.
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class EyeEnhancementDataset(Dataset):
    """Paired dataset: degraded smartphone-style -> slit-lamp quality."""

    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        patch_size: int = 256,
        augment: bool = True,
        is_train: bool = True,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.augment = augment and is_train
        self.is_train = is_train

        # Find paired files (filenames must match in both directories)
        input_files = {f.name for f in self.input_dir.glob("*.png")}
        target_files = {f.name for f in self.target_dir.glob("*.png")}
        paired = sorted(input_files & target_files)

        if not paired:
            raise FileNotFoundError(
                f"No matching .png pairs found between:\n"
                f"  input:  {self.input_dir}\n"
                f"  target: {self.target_dir}"
            )

        self.filenames = paired
        self.to_tensor = transforms.ToTensor()  # [0, 1] range

        print(f"{'Train' if is_train else 'Val'} dataset: {len(self.filenames)} paired images")

    def __len__(self):
        return len(self.filenames)

    def _random_crop(self, inp, tgt):
        """Random crop both images identically."""
        _, h, w = inp.shape
        ps = self.patch_size

        if h < ps or w < ps:
            # If image is smaller than patch, resize up
            inp = transforms.functional.resize(inp, [ps, ps])
            tgt = transforms.functional.resize(tgt, [ps, ps])
            return inp, tgt

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        inp = inp[:, top:top + ps, left:left + ps]
        tgt = tgt[:, top:top + ps, left:left + ps]
        return inp, tgt

    def _augment(self, inp, tgt):
        """Apply identical random augmentations to both images."""
        # Random horizontal flip
        if random.random() > 0.5:
            inp = torch.flip(inp, [-1])
            tgt = torch.flip(tgt, [-1])

        # Random vertical flip
        if random.random() > 0.5:
            inp = torch.flip(inp, [-2])
            tgt = torch.flip(tgt, [-2])

        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            inp = torch.rot90(inp, k, [-2, -1])
            tgt = torch.rot90(tgt, k, [-2, -1])

        return inp, tgt

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        inp_img = Image.open(self.input_dir / fname).convert("RGB")
        tgt_img = Image.open(self.target_dir / fname).convert("RGB")

        inp = self.to_tensor(inp_img)  # (3, H, W) in [0, 1]
        tgt = self.to_tensor(tgt_img)

        # Ensure same spatial size
        if inp.shape != tgt.shape:
            min_h = min(inp.shape[1], tgt.shape[1])
            min_w = min(inp.shape[2], tgt.shape[2])
            inp = inp[:, :min_h, :min_w]
            tgt = tgt[:, :min_h, :min_w]

        if self.is_train:
            inp, tgt = self._random_crop(inp, tgt)
            if self.augment:
                inp, tgt = self._augment(inp, tgt)
        else:
            # For validation, center-crop or resize to patch_size
            _, h, w = inp.shape
            ps = self.patch_size
            if h > ps or w > ps:
                top = (h - ps) // 2
                left = (w - ps) // 2
                inp = inp[:, top:top + ps, left:left + ps]
                tgt = tgt[:, top:top + ps, left:left + ps]
            elif h < ps or w < ps:
                inp = transforms.functional.resize(inp, [ps, ps])
                tgt = transforms.functional.resize(tgt, [ps, ps])

        return inp, tgt, fname


def create_train_val_split(
    input_dir: str,
    target_dir: str,
    val_ratio: float = 0.1,
    patch_size: int = 256,
    seed: int = 42,
):
    """Create train/val datasets with a random split."""
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)

    input_files = {f.name for f in input_dir.glob("*.png")}
    target_files = {f.name for f in target_dir.glob("*.png")}
    paired = sorted(input_files & target_files)

    random.seed(seed)
    random.shuffle(paired)
    n_val = max(1, int(len(paired) * val_ratio))
    val_files = set(paired[:n_val])

    # Write split files so it's reproducible
    split_dir = input_dir.parent / "splits"
    split_dir.mkdir(exist_ok=True)
    (split_dir / "train.txt").write_text("\n".join(sorted(set(paired) - val_files)))
    (split_dir / "val.txt").write_text("\n".join(sorted(val_files)))

    train_ds = EyeEnhancementDataset(input_dir, target_dir, patch_size, augment=True, is_train=True)
    val_ds = EyeEnhancementDataset(input_dir, target_dir, patch_size, augment=False, is_train=False)

    # Override the file lists
    train_ds.filenames = sorted(set(paired) - val_files)
    val_ds.filenames = sorted(val_files)

    print(f"Split: {len(train_ds)} train, {len(val_ds)} val")
    return train_ds, val_ds
