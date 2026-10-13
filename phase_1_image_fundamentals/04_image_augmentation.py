
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

# 3. Rotate the image by a random angle between -5° and +5°
import random

angle = random.uniform(-5, 5)

rotated_image = image.rotate(
    angle,
    resample=Image.Resampling.BICUBIC,
    expand=False,
    fillcolor=0
)

print("Rotation angle:", round(angle, 2))


# 4. Compare original and rotated images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Chest X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rotated_image, cmap="gray", vmin=0, vmax=255)
plt.title("Rotated Chest X-ray ({}°)".format(round(angle, 2)))
plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. RANDOM BRIGHTNESS AUGMENTATION
# ------------------------------------------------------------

import random
import numpy as np

# Random brightness adjustment between -20 and +20
brightness_value = random.randint(-20, 20)

# Convert the original X-ray to a NumPy array
image_array = np.array(image)

# Apply brightness adjustment
random_bright_image = np.clip(
    image_array.astype(np.int16) + brightness_value,
    0,
    255
).astype(np.uint8)

print("Brightness adjustment:", brightness_value)

# Compare original and augmented X-rays
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap="gray", vmin=0, vmax=255)
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(random_bright_image, cmap="gray", vmin=0, vmax=255)
plt.title(f"Brightness Adjustment ({brightness_value:+d})")
plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. COMBINED AUGMENTATION
#    Random rotation + random brightness
# ------------------------------------------------------------

# Select random transformation values
rotation_angle = random.uniform(-5, 5)
brightness_change = random.randint(-20, 20)

# First: Rotate the original X-ray
combined_rotated = image.rotate(
    rotation_angle,
    resample=Image.Resampling.BICUBIC,
    expand=False,
    fillcolor=0
)

# Second: Convert the rotated image into a NumPy array
combined_array = np.array(combined_rotated)

# Third: Apply brightness adjustment
combined_augmented = np.clip(
    combined_array.astype(np.int16) + brightness_change,
    0,
    255
).astype(np.uint8)

# Display selected transformations
print("\n===== COMBINED AUGMENTATION =====")
print("Rotation angle:", round(rotation_angle, 2))
print("Brightness adjustment:", brightness_change)

# Compare original and augmented X-rays
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(combined_augmented, cmap="gray", vmin=0, vmax=255)
plt.title("Rotation + Brightness Augmentation")
plt.axis("off")

plt.tight_layout()
plt.show()
