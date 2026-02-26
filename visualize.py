import cv2
import numpy as np
import os
from pathlib import Path

# ==========================================================
# 1. PATHS SETUP
# ==========================================================
base_dir = Path(__file__).resolve().parent.parent
mask_folder = base_dir / "TEST_RESULTS"
orig_folder = base_dir / "Offroad_Segmentation_testImages" / "Color_Images"
output_vis = base_dir / "Offroad_Segmentation_Scripts" / "Presentation_Visuals_Final"

os.makedirs(output_vis, exist_ok=True)

PALETTE = {
    0: (0, 0, 0), 1: (34, 139, 34), 2: (0, 255, 0), 3: (140, 180, 210), 4: (43, 90, 139),
    5: (0, 128, 128), 6: (19, 69, 139), 7: (128, 128, 128), 8: (45, 82, 160), 9: (235, 206, 135)
}

print(f"🚀 Starting Final Visualization...")

mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(('.png', '.jpg'))]

for filename in mask_files:
    # 1. Load Mask (Keep it as is first)
    mask_path = os.path.join(mask_folder, filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # 2. Fix Filename logic
    actual_name = filename.replace("res_", "")
    name_no_ext = os.path.splitext(actual_name)[0]

    # 3. Load Original
    orig = None
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
        temp_path = os.path.join(orig_folder, name_no_ext + ext)
        orig = cv2.imread(temp_path)
        if orig is not None: break

    if mask is None or orig is None:
        continue

    # --- THE CRITICAL FIX: Ensure SHAPES match perfectly ---
    # Hum pehle image ki height, width nikaalte hain
    h, w = orig.shape[:2]
    # Mask ko image ke exact size par resize karte hain
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Agar mask abhi bhi 3-channel hai (kabhi kabhi cv2 load kar leta hai), use single channel karo
    if len(mask_resized.shape) == 3:
        mask_resized = mask_resized[:, :, 0]

    # 4. Create Color Mask with exact same shape as 'orig'
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in PALETTE.items():
        # Masking assignment
        color_mask[mask_resized == class_id] = color

    # 5. Visual Effects
    overlay = cv2.addWeighted(orig, 0.6, color_mask, 0.4, 0)

    # Horizontal Stack
    final_strip = np.hstack((orig, color_mask, overlay))

    # Text
    cv2.putText(final_strip, f"ID: {name_no_ext}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # 6. Save
    cv2.imwrite(os.path.join(output_vis, f"final_{name_no_ext}.jpg"), final_strip)

print(f"SUCCESS! Check: {output_vis}")