import cv2
import os
import numpy as np
import pandas as pd
import json
import albumentations as A
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_IMG_DIR = "Data_Raw/Original_Slit-lamp_Images"
CSV_PATH = "Data_Raw/Annotations.csv"
OUTPUT_IMG_DIR = "Data_Degraded/images"
OUTPUT_MASK_DIR = "Data_Degraded/masks"
IMG_SIZE = 1024

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# --- CLASS MAPPING (Text -> Integer) ---
# Adjust these based on the exact spelling in your 'attributes' column
# Mapped from your Annotations.csv 'attributes' column
# --- CLASS MAPPING ---
CLASS_MAP = {
    "Background": 0,
    "Conjunctiva": 1,
    "Cornea": 2,
    "Pupil": 3,
    "Lesion": 4,      # Maps "Keratitis", "Cataract" (when standalone), etc.
    "Flash": 5        # If you have specular highlights
}

# --- HELPER: SMART RESIZE ---
def smart_resize(image, mask, target_size=1024):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Nearest neighbor is crucial for masks to keep integer class IDs (prevent 3 becoming 2.5)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    final_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    final_mask = np.zeros((target_size, target_size), dtype=np.uint8)

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    final_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_image
    final_mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_mask

    return final_image, final_mask

def generate_mask_from_csv(filename, image_shape, df):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    rows = df[df['filename'] == filename]
    if rows.empty:
        return mask

    parsed_shapes = []
    
    for idx, row in rows.iterrows():
        try:
            attr = json.loads(row['attributes'])
            shape = json.loads(row['shape_coordinates'])
            
            # --- FIX 1: Handle Empty Regions (Lesions) ---
            region_name = attr.get('region', '')
            lesion_name = attr.get('lesion', '')

            # If region is empty but lesion exists, treat as 'Lesion' class
            if region_name == '' and lesion_name != '':
                region_name = 'Lesion'
            
            class_id = CLASS_MAP.get(region_name, 0)
            if class_id == 0: continue # Skip background or unknown classes

            # --- FIX 2: Calculate Area for Sorting ---
            area = 0
            name = shape['name']
            if name == 'circle':
                area = 3.14159 * (shape['r'] ** 2)
            elif name == 'ellipse':
                # Use .get() to handle missing keys safely
                rx = shape.get('rx', 0)
                ry = shape.get('ry', 0)
                area = 3.14159 * rx * ry
            elif name == 'rect':
                area = shape['width'] * shape['height']
            elif name == 'polygon':
                area = h * w # Draw polygons first (usually background layers)
            
            parsed_shapes.append({
                'class_id': class_id,
                'shape': shape,
                'area': area
            })
            
        except Exception as e:
            print(f"Skipping row for {filename}: {e}")

    # Sort: Largest Area First (Background), Smallest Last (Foreground)
    parsed_shapes.sort(key=lambda x: x['area'], reverse=True)

    # --- DRAWING LOOP ---
    for item in parsed_shapes:
        shape = item['shape']
        class_id = int(item['class_id'])
        name = shape['name']

        if name == 'circle':
            cv2.circle(mask, (int(shape['cx']), int(shape['cy'])), int(shape['r']), class_id, -1)
        
        elif name == 'ellipse':
            # --- THE FIX IS HERE ---
            center = (int(shape['cx']), int(shape['cy']))
            axes = (int(shape['rx']), int(shape['ry']))
            # Default to 0 if 'theta' is missing
            angle = int(shape.get('theta', 0)) 
            cv2.ellipse(mask, center, axes, angle, 0, 360, class_id, -1)

        elif name == 'rect':
            x, y, w_rect, h_rect = int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height'])
            cv2.rectangle(mask, (x, y), (x+w_rect, y+h_rect), class_id, -1)

        elif name == 'polygon':
            pts = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), dtype=np.int32)
            cv2.fillPoly(mask, [pts], class_id)

    return mask

# --- DEGRADATION PIPELINE ---
degradation_pipeline = A.Compose([
    # 1. Resolution & Lens Quality (Simulate digital zoom/cheap lens)
    A.Downscale(scale_min=0.25, scale_max=0.5, p=0.5), # Aggressive downscale is key for "bad" phone look
    
    # 2. Blur (Mix of Focus miss and Hand shake)
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=(3, 7), p=0.5),  # <--- Added Motion Blur
    ], p=0.5),

    # 3. Noise (Mix of Sensor noise and High ISO grain)
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # <--- Fixed values for uint8
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5), # <--- Added ISO Noise
    ], p=0.5),

    # 4. Exposure & White Balance (Smartphones often struggle here)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

    # 5. Compression (The final smartphone save step)
    A.ImageCompression(quality_range=(30, 60), p=0.5), # Lowered quality slightly for more realism
])



# --- EXECUTION ---
print("Loading Annotations...")
df = pd.read_csv(CSV_PATH)
print("CSV Loaded. Columns found:", df.columns)

img_paths = sorted(glob(os.path.join(INPUT_IMG_DIR, "*.png"))) # Updated to .png based on your CSV

print(f"Processing {len(img_paths)} images...")

for img_path in tqdm(img_paths):
    filename = os.path.basename(img_path)
    
    # 1. Load Image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {filename}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Generate Mask from CSV
    mask = generate_mask_from_csv(filename, image.shape, df)

    # 3. Apply Smart Resize
    image_1024, mask_1024 = smart_resize(image, mask, target_size=1024)

    # 4. Apply Degradations
    transformed = degradation_pipeline(image=image_1024, mask=mask_1024)
    deg_image = transformed["image"]
    deg_mask = transformed["mask"]

    # 5. Save
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, filename), cv2.cvtColor(deg_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, filename), deg_mask)

print("Pipeline Complete.")
test_mask = cv2.imread("Data_Degraded/masks/1.png", cv2.IMREAD_UNCHANGED)
print(f"Unique values in mask: {np.unique(test_mask)}")