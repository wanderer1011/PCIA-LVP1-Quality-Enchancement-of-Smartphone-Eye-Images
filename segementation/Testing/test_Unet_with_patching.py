import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# --- CONFIG ---
ENCODER = 'efficientnet-b0'
CLASSES = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "../eye_segmentation_unet.pth"
TEST_IMAGE_PATH = "../Raw_Smartphone_Images/image_0007.jpg" # Update this to an actual image path!

# Define a color map for visualization (RGB)
# 0: Background (Black)
# 1: Conjunctiva (Red)
# 2: Cornea (Green)
# 3: Pupil (Blue)
# 4: Lesion (Yellow)
COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0]
}

def decode_mask(mask_2d, color_map):
    """Converts a 2D class mask into an RGB image for visualization."""
    rgb_mask = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        rgb_mask[mask_2d == class_idx] = color
    return rgb_mask

def predict_with_patches(image_path, model, device='cuda'):
    # 1. Load the original 1024x1024 image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ensure it's exactly 1024x1024 (Resize just in case your raw data varies slightly)
    image = cv2.resize(image, (1024, 1024))
    original_image = image.copy()

    # 2. Extract the 4 Patches (512x512 each)
    patches = [
        image[0:512, 0:512, :],       # Top-Left
        image[0:512, 512:1024, :],    # Top-Right
        image[512:1024, 0:512, :],    # Bottom-Left
        image[512:1024, 512:1024, :]  # Bottom-Right
    ]

    # 3. Preprocess the patches and batch them
    processed_patches = []
    for p in patches:
        p = (p / 255.0).astype(np.float32)
        p = np.transpose(p, (2, 0, 1)) # HWC -> CHW
        processed_patches.append(torch.tensor(p))

    # Stack into a single batch: Shape will be [4, 3, 512, 512]
    batch_tensor = torch.stack(processed_patches).to(device)

    # 4. Perform Inference on the batch
    with torch.no_grad():
        logits = model(batch_tensor) # Shape: [4, 5, 512, 512]

    # 5. Create an empty canvas to stitch the 1024x1024 logits back together
    # Shape: [1, 5, 1024, 1024]
    full_logits = torch.zeros((1, 5, 1024, 1024), device=device)

    # 6. Paste the predicted logits into their respective locations
    full_logits[0, :, 0:512, 0:512]       = logits[0]
    full_logits[0, :, 0:512, 512:1024]    = logits[1]
    full_logits[0, :, 512:1024, 0:512]    = logits[2]
    full_logits[0, :, 512:1024, 512:1024] = logits[3]

    # 7. Apply Argmax to the stitched 1024x1024 canvas
    pred_mask = torch.argmax(full_logits, dim=1).squeeze(0).cpu().numpy()

    return original_image, pred_mask

def test_model():
    print(f"Using Device: {DEVICE}")

    # 1. Initialize Model Architecture
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=None, # We are loading our own weights now
        in_channels=3, 
        classes=CLASSES,
        activation=None 
    )
    
    # 2. Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode (crucial for BatchNorm/Dropout)
    print("Model loaded successfully!")

    # 3. Perform Patch-based Inference
    original_image, pred_mask = predict_with_patches(TEST_IMAGE_PATH, model, DEVICE)

    # 4. Visualize the Results
    rgb_pred_mask = decode_mask(pred_mask, COLOR_MAP)
    
    # Create an overlay (both original_image and rgb_pred_mask are now 1024x1024)
    overlay = cv2.addWeighted(original_image, 0.6, rgb_pred_mask, 0.4, 0)

    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image (1024x1024)")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Stitched Predicted Mask")
    plt.imshow(rgb_pred_mask)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_model()