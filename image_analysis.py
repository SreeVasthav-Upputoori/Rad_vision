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

# Increase brightness by 50
bright_image = np.clip(
    image_array.astype(np.int16) + 50,
    0,
    255
).astype(np.uint8)
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

plt.figure(figsize=(12, 5))

# Original X-ray
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap="gray", vmin=0, vmax=255)
plt.title("Original X-ray")
plt.axis("off")

# Brightened X-ray
plt.subplot(1, 2, 2)
plt.imshow(bright_image, cmap="gray", vmin=0, vmax=255)
plt.title("Brightness +50")
plt.axis("off")

plt.tight_layout()
plt.show()
