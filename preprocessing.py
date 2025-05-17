import os
import cv2
import json
import math
from collections import Counter
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

"""
Hand image preprocessing & augmentation pipeline for ASL letter recognition (ResNet‑18).
"""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIRS = ["data/train", "data/test"]
OUTPUT_BASE = "data_processed"
FLIPPABLE_CLASSES = {
    # ASL letters whose meaning is orientation‑invariant
    "A", "B", "C", "E", "M", "N", "O", "S", "T", "U", "V", "W", "X"
}
# J & Z have been removed.

# -----------------------------------------------------------------------------
# Permanent preprocessing (disk)
# -----------------------------------------------------------------------------
PREPROCESS = transforms.Compose([
    transforms.Resize((128, 128)),  # idempotent safeguard
    transforms.Grayscale(num_output_channels=1),
])

# -----------------------------------------------------------------------------
# Custom transforms
# -----------------------------------------------------------------------------
class AddGaussianNoise:
    """Additive zero‑mean Gaussian noise."""
    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, img: torch.Tensor):  # expects tensor in [0,1]
        noise = torch.randn_like(img) * self.sigma
        return torch.clamp(img + noise, 0.0, 1.0)

class ClassAwareRandomHorizontalFlip:
    """Flip only if target class is in FLIPPABLE_CLASSES."""
    def __init__(self, p: float = 0.5, flippable=FLIPPABLE_CLASSES):
        self.p = p
        self.flippable = flippable
        self.hflip = transforms.functional.hflip

    def __call__(self, img, target):
        if target in self.flippable and torch.rand(1).item() < self.p:
            img = self.hflip(img)
        return img

# -----------------------------------------------------------------------------
# Online augmentation (used in training DataLoader)
# -----------------------------------------------------------------------------
AUGMENT_TRAIN = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    AddGaussianNoise(sigma=0.03),
])

# NORMALIZE will be set after stats computation
NORMALIZE = None

# -----------------------------------------------------------------------------
# Image cleaning helpers
# -----------------------------------------------------------------------------

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Return denoised, CLAHE‑enhanced single‑channel image.

    Accepts BGR or already‑grayscale images (2‑D or 3‑D with 1 channel).
    """
    # Detect channel count & convert to gray only when needed
    if len(img_bgr.shape) == 2:  # already (H,W)
        gray = img_bgr
    elif img_bgr.shape[2] == 1:  # (H,W,1)
        gray = img_bgr[:, :, 0]
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE for lighting normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light Gaussian blur to suppress noise
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return denoised

# -----------------------------------------------------------------------------
# Disk preprocessing loop + mean/std accumulation
# -----------------------------------------------------------------------------

def process_dataset(input_dir: str, output_dir: str) -> Tuple[float, float, int]:
    """Process a directory; return (sum, sum_sq, pixel_count)."""
    ch_sum = 0.0
    ch_sum_sq = 0.0
    num_pixels = 0

    for class_name in os.listdir(input_dir):
        src_dir = os.path.join(input_dir, class_name)
        dst_dir = os.path.join(output_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        img_files = [f for f in os.listdir(src_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fname in tqdm(img_files, desc=f"{class_name:>12}"):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".png")
            try:
                img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                processed = preprocess_image(img)
                pil_img = Image.fromarray(processed)
                pil_img = PREPROCESS(pil_img)
                pil_img.save(dst_path)

                # accumulate stats (tensor already grayscale 0‑255)
                arr = torch.from_numpy(np.array(pil_img)).float() / 255.0
                ch_sum += arr.sum().item()
                ch_sum_sq += (arr ** 2).sum().item()
                num_pixels += arr.numel()
            except Exception as e:
                print(f"[ERROR] {src_path}: {e}")
    return ch_sum, ch_sum_sq, num_pixels

# -----------------------------------------------------------------------------
# Helpers for training to create balanced sampler
# -----------------------------------------------------------------------------

def create_balanced_sampler(dataset):
    """Return a WeightedRandomSampler balancing class frequencies."""
    targets = dataset.targets  # expects torchvision‑style dataset
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[t] for t in targets]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    global NORMALIZE

    # --- train split preprocessing & stats ---
    train_dir = [p for p in INPUT_DIRS if p.endswith("train")][0]
    out_train = os.path.join(OUTPUT_BASE, "train")
    s_sum, s_sum_sq, s_pixels = process_dataset(train_dir, out_train)

    mean = s_sum / s_pixels
    std = math.sqrt(max(s_sum_sq / s_pixels - mean ** 2, 1e-8))
    NORMALIZE = transforms.Normalize(mean=[mean], std=[std])

    stats_path = os.path.join(OUTPUT_BASE, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    print(f"Saved dataset stats to {stats_path} → mean={mean:.4f}, std={std:.4f}")

    # --- test split ---
    test_dir = [p for p in INPUT_DIRS if p.endswith("test")][0]
    out_test = os.path.join(OUTPUT_BASE, "test")
    process_dataset(test_dir, out_test)

if __name__ == "__main__":
    main()
