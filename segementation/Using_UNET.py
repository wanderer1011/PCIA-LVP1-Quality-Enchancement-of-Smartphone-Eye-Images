import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# --- CONFIG ---
DATA_DIR = "Data Generation/Data_Degraded"
CLASSES = 5  # 0:Back, 1:Conjunctiva, 2:Cornea, 3:Pupil, 4:Lesion
ENCODER = 'efficientnet-b0' # Lightweight and fast
WEIGHTS = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {DEVICE}")

if DEVICE == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- DATASET ---
class EyeDataset(Dataset):
    def __init__(self, root_dir, augmentation=None):
        self.root_dir = root_dir
        self.images = sorted(os.listdir(os.path.join(root_dir, "images")))
        self.masks = sorted(os.listdir(os.path.join(root_dir, "masks")))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load Image
        img_path = os.path.join(self.root_dir, "images", self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])
        mask = cv2.imread(mask_path, 0) # Read as grayscale (0-4)
        
        # Resize to closest multiple of 32 (Requirement for U-Net)
        # For simplicity, let's just resize to 512x512 here
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Normalize Image to 0-1 and swap axes (HWC -> CHW)
        image = (image / 255.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        
        # Mask needs to be LongTensor for CrossEntropy
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        return torch.tensor(image), mask

# --- MODEL & TRAINING ---
def train():
    # 1. Create Model (The Magic Part)
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=WEIGHTS, 
        in_channels=3, 
        classes=CLASSES,
        activation=None # We use CrossEntropyLoss, so raw logits needed
    )
    model.to(DEVICE)

    # 2. Setup Data
    dataset = EyeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. Optimizer & Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 4. Loop
    print("Starting training...")
    for epoch in range(10): # Run for more epochs in reality
        total_loss = 0
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images) # Output shape: [Batch, 5, 512, 512]
            
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "eye_segmentation_unet.pth")
    print("Saved!")

if __name__ == "__main__":
    train()