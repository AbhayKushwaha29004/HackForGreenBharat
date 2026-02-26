import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# ============================================================================
# CONFIGURATION & GPU
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
n_classes = len(value_map)


# ============================================================================
# UTILS & METRICS
# ============================================================================
def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr


def compute_iou(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    # Use SMP metrics for faster and accurate calculation
    tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=10)
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()


# ============================================================================
# DATASET CLASS
# ============================================================================
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = cv2.imread(os.path.join(self.image_dir, data_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        return image, mask


# ============================================================================
# MODEL HEAD (ConvNeXt-style)
# ============================================================================
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    print(f"\nUsing Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Configuration
    batch_size = 8  # 3050 Ti ke liye sweet spot
    w, h = 448, 252  # DINOv2 requires multiple of 14
    lr = 5e-5
    n_epochs = 40
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Transforms
    train_transform = A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Dataloaders
    train_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    train_loader = DataLoader(MaskDataset(train_dir, train_transform), batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(MaskDataset(val_dir, val_transform), batch_size=batch_size, shuffle=False)

    # Backbone & Head
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14").to(DEVICE).eval()
    classifier = SegmentationHeadConvNeXt(in_channels=384, out_channels=10, tokenW=w // 14, tokenH=h // 14).to(DEVICE)

    # Combo Loss & Optimizer
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_iou = 0.0

    print("\n🔥 Starting Pro-Level Training...")
    for epoch in range(n_epochs):
        classifier.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")

        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]

            with torch.amp.autocast(device_type='cuda'):
                logits = classifier(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                loss = 0.5 * dice_loss(outputs, labels) + 0.5 * focal_loss(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- VALIDATION ---
        classifier.eval()
        val_iou = 0
        with torch.no_grad():
            for v_imgs, v_labels in val_loader:
                v_imgs, v_labels = v_imgs.to(DEVICE), v_labels.to(DEVICE)
                v_feat = backbone.forward_features(v_imgs)["x_norm_patchtokens"]
                v_logits = classifier(v_feat)
                v_outputs = F.interpolate(v_logits, size=v_imgs.shape[2:], mode="bilinear", align_corners=False)
                val_iou += compute_iou(v_outputs, v_labels)

        avg_val_iou = val_iou / len(val_loader)
        scheduler.step()

        print(
            f"📊 Epoch {epoch + 1} Summary -> Train Loss: {train_loss / len(train_loader):.4f}, Val IoU: {avg_val_iou:.4f}")

        # --- BEST MODEL SAVER ---
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head.pth"))
            print(f"🌟 New Best IoU: {best_iou:.4f}! Saved to segmentation_head.pth")


if __name__ == "__main__":
    main()