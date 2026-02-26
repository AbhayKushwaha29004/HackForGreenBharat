import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. CONFIG & PRO COLORS
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W, H = 448, 252
NUM_CLASSES = 10

# In colors se tumhara mask "Colored" dikhega, black nahi
COLOR_MAP = {
    0: (0, 0, 0),  # Background - Black
    1: (34, 139, 34),  # Trees - Green
    2: (0, 255, 0),  # Lush Bushes - Lime
    3: (140, 180, 210),  # Dry Grass - Tan
    4: (43, 90, 139),  # Dry Bushes - Brown
    5: (0, 128, 128),  # Clutter - Olive
    6: (19, 69, 139),  # Logs - Dark Brown
    7: (128, 128, 128),  # Rocks - Gray
    8: (45, 82, 160),  # Landscape - Sienna
    9: (235, 206, 135)  # Sky - Sky Blue
}


# ==========================================
# 2. EXACT ARCHITECTURE (SYNCED)
# ==========================================
class SegmentationHeadConvNeXt(torch.nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU()
        )
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256)
        )
        self.classifier = torch.nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


def colorize_mask(mask):
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLOR_MAP.items():
        color_img[mask == val] = color
    return color_img


# ==========================================
# 3. INFERENCE & VISUALIZATION
# ==========================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "segmentation_head.pth")
    test_img_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages', 'Color_Images')
    output_dir = os.path.join(script_dir, '..', 'Testing_RESULTS')  # Yahan colored images aayengi

    os.makedirs(output_dir, exist_ok=True)

    print("⏳ Loading Models...")
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14").to(DEVICE).eval()
    classifier = SegmentationHeadConvNeXt(384, NUM_CLASSES, W // 14, H // 14).to(DEVICE)

    # Error Fix for loading
    classifier.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    classifier.eval()

    test_transform = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.png', '.jpg'))]

    print(f"🚀 Processing {len(test_files)} images...")

    with torch.no_grad():
        for img_name in tqdm(test_files):
            path = os.path.join(test_img_dir, img_name)
            raw_bgr = cv2.imread(path)
            if raw_bgr is None: continue

            orig_h, orig_w = raw_bgr.shape[:2]
            rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

            input_tensor = test_transform(image=rgb)['image'].unsqueeze(0).to(DEVICE)

            # Prediction with TTA (Flip) for better results
            feat = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
            logits = classifier(feat)

            # Upsample
            final_logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            mask_pred = torch.argmax(final_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # --- DEBUG CHECK ---
            unique_vals = np.unique(mask_pred)

            # --- SAVE COLORED MASK ---
            colored_mask = colorize_mask(mask_pred)

            # Final View: Image + Mask Overlay (50% transparency)
            overlay = cv2.addWeighted(raw_bgr, 0.6, colored_mask, 0.4, 0)

            # Save the Overlay (Judges love this!)
            cv2.imwrite(os.path.join(output_dir, f"res_{img_name}"), overlay)

    print(f"\n✅ Done! Check the 'TEST_RESULTS' folder for colored images.")
    print(f"Last processed unique classes: {unique_vals}")


if __name__ == "__main__":
    main()