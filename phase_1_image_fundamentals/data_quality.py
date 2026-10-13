
# ============================================================
# RADVISION AI — PHASE 1
# FILE 06: DATASET QUALITY CHECKS
# ============================================================
# Lesson 12 — Checking for corrupted image files
# ============================================================

from pathlib import Path
from PIL import Image

# Dataset location
dataset_path = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney"
    r"\chest-xray-pneumonia\versions\2"
    r"\chest_xray"
)

splits = ["train", "val", "test"]
categories = ["NORMAL", "PNEUMONIA"]

corrupted_images = []
total_checked = 0

# Check every image
for split in splits:
    for category in categories:

        folder = dataset_path / split / category

        for image_path in folder.glob("*.jpeg"):

            total_checked += 1

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except (OSError, ValueError) as error:
                corrupted_images.append(str(image_path))
                print("Corrupted image:", image_path.name)
                print("Error:", error)

# Display results
print("\n===== DATASET QUALITY REPORT =====")
print("Total images checked:", total_checked)
print("Corrupted images:", len(corrupted_images))
