
# ============================================================
# RADVISION AI — PHASE 1
# FILE 05: DATASET EXPLORATION
# ============================================================
# Objective: Count NORMAL and PNEUMONIA X-rays
# in training, validation, and testing datasets.
# ============================================================

from pathlib import Path

# 1. Dataset location
dataset_path = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney"
    r"\chest-xray-pneumonia\versions\2"
    r"\chest_xray"
)

# 2. Dataset categories
splits = ["train", "val", "test"]
categories = ["NORMAL", "PNEUMONIA"]

# 3. Count images in each category
for split in splits:

    print(f"\n===== {split.upper()} DATASET =====")

    for category in categories:

        folder = dataset_path / split / category

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

        image_count = len(list(folder.glob("*.jpeg")))

        print(f"{category}: {image_count} images")

# ------------------------------------------------------------
# 4. CALCULATE CLASS DISTRIBUTION PERCENTAGES
# ------------------------------------------------------------

for split in splits:

    print(f"\n===== {split.upper()} CLASS DISTRIBUTION =====")

    class_counts = {}

    # Count images in each class
    for category in categories:
        folder = dataset_path / split / category
        class_counts[category] = len(list(folder.glob("*.jpeg")))

    # Calculate total images
    total_images = sum(class_counts.values())

    # Calculate class percentages
    for category, count in class_counts.items():
        percentage = (count / total_images) * 100

        print(f"{category}: {count} images ({percentage:.2f}%)")

# ------------------------------------------------------------
# 5. VISUALISE SAMPLE IMAGES AND LABELS
# ------------------------------------------------------------

from PIL import Image
import matplotlib.pyplot as plt

# Numerical labels for our classification model
class_labels = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

plt.figure(figsize=(10, 9))

plot_position = 1

for category in categories:

    # Select images from the training dataset
    folder = dataset_path / "train" / category

    # Select the first two JPEG images
    image_files = sorted(folder.glob("*.jpeg"))[:2]

    for image_path in image_files:

        # Load image in grayscale
        image = Image.open(image_path).convert("L")

        # Display in a 2x2 grid
        plt.subplot(2, 2, plot_position)
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)

        plt.title(
            f"{category} (Label: {class_labels[category]})"
        )

        plt.axis("off")

        plot_position += 1

plt.tight_layout()
plt.show()