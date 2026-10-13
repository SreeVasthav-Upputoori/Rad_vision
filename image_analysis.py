from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Location of NORMAL training images
folder = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney"
    r"\chest-xray-pneumonia\versions\2"
    r"\chest_xray\train\NORMAL"
)

# Find a JPEG image
image_path = next(folder.glob("*.jpeg"))

# Open the image
image = Image.open(image_path)

# Convert to grayscale
image = image.convert("L")

# Convert into NumPy array
image_array = np.array(image)
# Normalise pixel intensities from 0–255 to 0–1
normalized_image = image_array.astype(np.float32) / 255.0

# Increase brightness by 50
bright_image = np.clip(
    image_array.astype(np.int16) + 50,
    0,
    255
).astype(np.uint8)
# Display the original and brightened images
print("Pixel at (100, 200):", image_array[100, 200])
print("Minimum intensity:", image_array.min())
print("Maximum intensity:", image_array.max())
print("Mean intensity:", image_array.mean())
print("Pixel at (100, 200) in bright image:", bright_image[100, 200])
print("Minimum intensity in bright image:", bright_image.min())
print("Maximum intensity in bright image:", bright_image.max())
print("Mean intensity in bright image:", bright_image.mean())

# Inspect
print("Image:", image_path.name)
print("Dimensions:", image_array.shape)
print("Data type:", image_array.dtype)

# Increase contrast by 50%
contrast_factor = 1.5

contrast_image = np.clip(
    (image_array.astype(np.float32) - 128)
    * contrast_factor + 128,
    0,
    255
).astype(np.uint8)

print("Original image:")
print("Minimum:", image_array.min())
print("Maximum:", image_array.max())
print("Mean:", image_array.mean())
print("Data type:", image_array.dtype)

print("\nNormalised image:")
print("Minimum:", normalized_image.min())
print("Maximum:", normalized_image.max())
print("Mean:", normalized_image.mean())
print("Data type:", normalized_image.dtype)

plt.figure(figsize=(12, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap="gray", vmin=0, vmax=255)
plt.title("Original X-ray")
plt.axis("off")

# Normalised image
plt.subplot(1, 2, 2)
plt.imshow(normalized_image, cmap="gray", vmin=0, vmax=1)
plt.title("Normalised X-ray (0–1)")
plt.axis("off")

plt.tight_layout()
plt.show()
