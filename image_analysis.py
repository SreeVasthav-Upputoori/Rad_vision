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
print("Pixel at (100, 200):", image_array[100, 200])
print("Minimum intensity:", image_array.min())
print("Maximum intensity:", image_array.max())
print("Mean intensity:", image_array.mean())

# Inspect
print("Image:", image_path.name)
print("Dimensions:", image_array.shape)
print("Data type:", image_array.dtype)

# Display
plt.imshow(image_array, cmap="gray")
plt.title("Normal Chest X-ray")
plt.axis("off")
plt.show()