
# ============================================================
# RADVISION AI — PHASE 1
# FILE 04: IMAGE AUGMENTATION
# ============================================================
# Lesson 10 — Experiment 1: Image Rotation
# ============================================================

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# 1. Dataset location
folder = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney"
    r"\chest-xray-pneumonia\versions\2"
    r"\chest_xray\train\NORMAL"
)


# 2. Load one X-ray
image_files = sorted(folder.glob("*.jpeg"))

if not image_files:
    raise FileNotFoundError("No JPEG images found.")

image_path = image_files[0]

image = Image.open(image_path).convert("L")


# 3. Rotate the image by 5 degrees
rotated_image = image.rotate(
    5,
    resample=Image.Resampling.BICUBIC,
    expand=False,
    fillcolor=0
)


# 4. Compare original and rotated images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Chest X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rotated_image, cmap="gray", vmin=0, vmax=255)
plt.title("Rotated Chest X-ray (+5°)")
plt.axis("off")

plt.tight_layout()
plt.show()
