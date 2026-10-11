from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

#Define path to the dataset folder.
DATASET_PATH = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray")

#Define class folders
normal_path = DATASET_PATH/"train"/"NORMAL"
pneumonia_path = DATASET_PATH/"train"/"PNEUMONIA"

#Get image files from each class folder.
normal_images = list(normal_path.iterdir())
pneumonia_images = list(pneumonia_path.iterdir())

#Select a sample from each class
normal_samples = normal_images[:5]  # Select the first 5 images from the NORMAL class
pneumonia_samples = pneumonia_images[:5]  # Select the first 5 images from the PNEUMONIA class

#create a grid
fig,axes = plt.subplots(2,5, figsize=(12,6))

#Display normal images
for i, img_path in enumerate(normal_samples):
    image = Image.open(img_path) #Open the image
    axes[0,i].imshow(image, cmap='gray') #Display the image in grayscale
    axes[0, i].set_title("Normal")
    axes[0,i].axis("off")

#Display pneumonia images
for i, img_path in enumerate(pneumonia_samples):
    image = Image.open(img_path) #Open the image
    axes[1,i].imshow(image, cmap='gray')
    axes[1, i].set_title("Pneumonia")
    axes[1,i].axis("off")

plt.tight_layout()

plt.show()