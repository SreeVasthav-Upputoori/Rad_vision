
# ============================================================
# RADVISION AI — PHASE 1
# FILE 02: BRIGHTNESS AND CONTRAST
# ============================================================
# Objectives:
# 1. Understand brightness adjustment
# 2. Understand contrast adjustment
# 3. Apply pixel clipping (0–255)
# 4. Compare original and modified images
# 5. Analyse changes using histograms
# ============================================================

from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. DATASET LOCATION
# ------------------------------------------------------------

folder = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney"
    r"\chest-xray-pneumonia\versions\2"
    r"\chest_xray\train\NORMAL"
)


# ------------------------------------------------------------
# 2. LOAD ONE CHEST X-RAY
# ------------------------------------------------------------

image_files = sorted(folder.glob("*.jpeg"))

if not image_files:
    raise FileNotFoundError("No JPEG images found.")

image_path = image_files[0]

image = Image.open(image_path).convert("L")
image_array = np.array(image)

print("Image:", image_path.name)


# ------------------------------------------------------------
# 3. BRIGHTNESS MODIFICATION
# ------------------------------------------------------------
# Formula:
# New intensity = Original intensity + brightness value
#
# np.clip() prevents values going below 0 or above 255.
# int16 prevents uint8 overflow during addition.

brightness_value = 50

bright_image = np.clip(
    image_array.astype(np.int16) + brightness_value,
    0,
    255
).astype(np.uint8)


# ------------------------------------------------------------
# 4. CONTRAST MODIFICATION
# ------------------------------------------------------------
# Formula:
# New intensity = (Original intensity - midpoint)
#                 * contrast_factor + midpoint
#
# Factor > 1 : Increase contrast
# Factor = 1 : No change
# Factor < 1 : Decrease contrast

contrast_factor = 1.5
midpoint = 128

contrast_image = np.clip(
    (image_array.astype(np.float32) - midpoint)
    * contrast_factor + midpoint,
    0,
    255
).astype(np.uint8)


# ------------------------------------------------------------
# 5. COMPARE PIXEL STATISTICS
# ------------------------------------------------------------

def print_statistics(name, array):
    print(f"\n===== {name} =====")
    print("Pixel [100, 200]:", array[100, 200])
    print("Minimum:", array.min())
    print("Maximum:", array.max())
    print("Mean:", array.mean())
    print("Standard deviation:", array.std())


print_statistics("ORIGINAL IMAGE", image_array)
print_statistics("BRIGHTENED IMAGE", bright_image)
print_statistics("CONTRAST-ADJUSTED IMAGE", contrast_image)


# ------------------------------------------------------------
# 6. DISPLAY ORIGINAL AND MODIFIED IMAGES
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

images = [
    (image_array, "Original X-ray"),
    (bright_image, "Brightness +50"),
    (contrast_image, "Contrast 1.5x")
]

for i, (img, title) in enumerate(images, start=1):
    plt.subplot(1, 3, i)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. COMPARE PIXEL INTENSITY HISTOGRAMS
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

for i, (img, title) in enumerate(images, start=1):
    plt.subplot(1, 3, i)
    plt.hist(img.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel("Pixel Intensity (0–255)")
    plt.ylabel("Number of Pixels")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. CALCULATE CLIPPED PIXELS
# ------------------------------------------------------------
# Count pixels where the transformation reaches either limit.
# This helps us understand information loss.

bright_clipped = np.sum(
    image_array.astype(np.int16) + brightness_value > 255
)

contrast_raw = (
    (image_array.astype(np.float32) - midpoint)
    * contrast_factor + midpoint
)

contrast_clipped = np.sum(
    (contrast_raw < 0) | (contrast_raw > 255)
)

print("\n===== CLIPPING ANALYSIS =====")
print("Pixels clipped by brightness:", bright_clipped)
print("Pixels clipped by contrast:", contrast_clipped)

print("Brightness clipping percentage:",
      round(bright_clipped / image_array.size * 100, 2), "%")

print("Contrast clipping percentage:",
      round(contrast_clipped / image_array.size * 100, 2), "%")
