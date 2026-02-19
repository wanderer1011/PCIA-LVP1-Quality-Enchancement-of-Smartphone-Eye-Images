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

    # 3. Load and Preprocess the Test Image
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not find image at {TEST_IMAGE_PATH}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy() # Keep a copy for visualization

    # Same preprocessing as in EyeDataset
    image = cv2.resize(image, (512, 512))
    image_tensor = (image / 255.0).astype(np.float32)
    image_tensor = np.transpose(image_tensor, (2, 0, 1)) # HWC -> CHW
    
    # Add batch dimension: [1, 3, 512, 512]
    image_tensor = torch.tensor(image_tensor).unsqueeze(0).to(DEVICE) 

    # 4. Perform Inference
    with torch.no_grad(): # Disable gradient calculation for memory efficiency
        logits = model(image_tensor)
        
        # logits shape is [1, 5, 512, 512]. Apply argmax across the class dimension (dim=1)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0) # Shape: [512, 512]
        pred_mask = pred_mask.cpu().numpy()

    # 5. Visualize the Results
    rgb_pred_mask = decode_mask(pred_mask, COLOR_MAP)

    # Resize original image to 512x512 for a side-by-side comparison
    original_resized = cv2.resize(original_image, (512, 512))

    # Optional: Create an overlay
    overlay = cv2.addWeighted(original_resized, 0.6, rgb_pred_mask, 0.4, 0)

    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_resized)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
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