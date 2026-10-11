#Data inspection script
# pathlib is used for working with folders and file paths.
from pathlib import Path

#Counter helps us to count images belongs to each class.
from collections import Counter

#PIL (Py Image Lib) allows us to open and inspect, manipulate imgaes.
from PIL import Image


#Define path to the dataset folder.
DATASET_PATH = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray"
)
#Datasets sturture
splits = ["train","test","val"]

classes = ["NORMAL","PNEUMONIA"]

#2. Count Images in each class for each split.
for split in splits:
    print("\n"+"="*20)
    print(split.upper())
    print("="*20)

    split_path = DATASET_PATH/split

    for class_name in classes:
        class_path = split_path / class_name  # Create path to the class folder.
        image_files = list(class_path.iterdir()) #count all files in the class folder.
        number_of_images = len(image_files) #Get the number of images in the class folder.
        print(f"{class_name}: {number_of_images}")

#3. Inspect some image properties.
print("\n"+"="*20)
print("IMAGE PROPERTIES")
print("="*20)

#Let'a take normal image from the training set and inspect its properties.
normal_path = DATASET_PATH/"train"/"NORMAL"
normal_images = list(normal_path.iterdir())
# Select the first image.

sample_image = normal_images[0]


# Open the image.

image = Image.open(sample_image)


# Print information about it.

print("\nSample image:")
print(sample_image.name)

print("\nImage size:")
print(image.size)

print("\nImage mode:")
print(image.mode)

