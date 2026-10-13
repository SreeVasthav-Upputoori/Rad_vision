# ============================================================
# RADVISION AI — PHASE 1
# FILE 01: IMAGE EXPLORATION
# ============================================================
# Objectives:
# 1. Load a chest X-ray
# 2. Convert it to grayscale
# 3. Convert it to a NumPy array
# 4. Inspect image dimensions and data type
# 5. Analyse individual pixel intensities
# 6. Visualise the image and its histogram
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

# Find all JPEG images
image_files = sorted(folder.glob("*.jpeg"))

if not image_files:
    raise FileNotFoundError("No JPEG images found in the folder.")

# Select the first image
image_path = image_files[0]

# Open image
image = Image.open(image_path)

# Convert to grayscale
image = image.convert("L")


# ------------------------------------------------------------
# 3. CONVERT IMAGE TO NUMPY ARRAY
# ------------------------------------------------------------

image_array = np.array(image)


# ------------------------------------------------------------
# 4. EXPLORE IMAGE PROPERTIES
# ------------------------------------------------------------

print("\n===== IMAGE INFORMATION =====")

print("Filename:", image_path.name)
print("Dimensions (Height, Width):", image_array.shape)
print("Data type:", image_array.dtype)
print("Total pixels:", image_array.size)


# ------------------------------------------------------------
# 5. PIXEL INTENSITY ANALYSIS
# ------------------------------------------------------------

print("\n===== PIXEL INTENSITIES =====")

print("Pixel at [100, 200]:", image_array[100, 200])
print("Minimum intensity:", image_array.min())
print("Maximum intensity:", image_array.max())
print("Mean intensity:", image_array.mean())
print("Standard deviation:", image_array.std())


# ------------------------------------------------------------
# 6. VISUALISE X-RAY AND HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

# Original X-ray
plt.subplot(1, 2, 1)

plt.imshow(image_array, cmap="gray", vmin=0, vmax=255)
plt.title("Original Chest X-ray")
plt.axis("off")


# Pixel intensity histogram
plt.subplot(1, 2, 2)

plt.hist(
    image_array.ravel(),
    bins=256,
    range=(0, 256)
)

plt.title("Pixel Intensity Histogram")
plt.xlabel("Pixel Intensity (0–255)")
plt.ylabel("Number of Pixels")

plt.tight_layout()
plt.show()