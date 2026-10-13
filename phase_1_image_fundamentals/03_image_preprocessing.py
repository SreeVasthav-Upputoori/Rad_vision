
# ============================================================
# RADVISION AI — PHASE 1
# FILE 03: IMAGE PREPROCESSING
# ============================================================
# Objectives:
# 1. Normalise pixel intensities (0–255 to 0–1)
# 2. Understand image resizing
# 3. Compare image dimensions and pixel values
# 4. Preserve aspect ratio during resizing
# 5. Visualise the preprocessing results
# ============================================================

from pathlib import Path
from PIL import Image, ImageOps
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
# 3. IMAGE NORMALISATION
# ------------------------------------------------------------
# Formula:
# Normalised intensity = Original intensity / 255
#
# Convert to float32 first to support decimal values.

normalized_image = image_array.astype(np.float32) / 255.0


# ------------------------------------------------------------
# 4. IMAGE RESIZING
# ------------------------------------------------------------
# Resize image directly to 224 x 224 pixels.
#
# Warning:
# Direct resizing can distort the anatomical proportions
# if the original image is not square.

target_size = (224, 224)

resized_image = image.resize(
    target_size,
    resample=Image.Resampling.LANCZOS
)

resized_array = np.array(resized_image)


# ------------------------------------------------------------
# 5. ASPECT-RATIO-PRESERVING RESIZING
# ------------------------------------------------------------
# Resize while keeping original anatomical proportions.
# Padding is added if necessary to obtain a 224 x 224 image.
#
# This example uses black padding.

resized_padded = ImageOps.pad(
    image,
    target_size,
    method=Image.Resampling.LANCZOS,
    color=0,
    centering=(0.5, 0.5)
)

padded_array = np.array(resized_padded)


# ------------------------------------------------------------
# 6. NORMALISE THE RESIZED IMAGE
# ------------------------------------------------------------
# This is a common preprocessing sequence:
# Resize -> Convert to NumPy -> Normalise

processed_image = padded_array.astype(np.float32) / 255.0


# ------------------------------------------------------------
# 7. PRINT IMAGE STATISTICS
# ------------------------------------------------------------

def print_statistics(name, array):
    print(f"\n===== {name} =====")
    print("Dimensions:", array.shape)
    print("Minimum:", array.min())
    print("Maximum:", array.max())
    print("Mean:", array.mean())
    print("Data type:", array.dtype)


print_statistics("ORIGINAL IMAGE", image_array)
print_statistics("NORMALISED IMAGE", normalized_image)
print_statistics("RESIZED IMAGE (224x224)", resized_array)
print_statistics("RESIZED WITH PADDING", padded_array)
print_statistics("RESIZED AND NORMALISED", processed_image)


# ------------------------------------------------------------
# 8. COMPARE ORIGINAL AND NORMALISED IMAGES
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap="gray", vmin=0, vmax=255)
plt.title("Original X-ray (0-255)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalized_image, cmap="gray", vmin=0, vmax=1)
plt.title("Normalised X-ray (0-1)")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 9. COMPARE RESIZING METHODS
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

images = [
    (image_array, "Original X-ray"),
    (resized_array, "Direct Resize (224x224)"),
    (padded_array, "Resize + Padding (224x224)")
]

for i, (img, title) in enumerate(images, start=1):
    plt.subplot(1, 3, i)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. COMPARE ORIGINAL AND NORMALISED HISTOGRAMS
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(image_array.ravel(), bins=256, range=(0, 256))
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Pixel Count")

plt.subplot(1, 2, 2)
plt.hist(normalized_image.ravel(), bins=256, range=(0, 1))
plt.title("Normalised Histogram")
plt.xlabel("Pixel Intensity (0-1)")
plt.ylabel("Pixel Count")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. FINAL OUTPUT INFORMATION
# ------------------------------------------------------------

print("\n===== PREPROCESSING COMPLETE =====")
print("Original dimensions:", image_array.shape)
print("Model input dimensions:", processed_image.shape)
print("Model input range:",
      processed_image.min(), "to", processed_image.max())
print("Model input data type:", processed_image.dtype)

# Compare original and resized image dimensions
print("\n===== RESIZING COMPARISON =====")

print("Original size:", image.size)
print("Direct resized size:", resized_image.size)
print("Padded resized size:", resized_padded.size)

print("Original aspect ratio:", image.width / image.height)
print("Direct resize aspect ratio:", resized_image.width / resized_image.height)