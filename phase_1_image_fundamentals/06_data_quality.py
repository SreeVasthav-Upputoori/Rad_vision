
# ============================================================
# RADVISION AI — PHASE 1
# FILE 06: DATASET QUALITY CHECKS
# ============================================================
# Lesson 12 — Checking for corrupted image files
# ============================================================

from pathlib import Path
from PIL import Image

# Dataset location
dataset_path = Path(
    r"D:\Tools\MED_IMG\datasets\paultimothymooney"
    r"\chest-xray-pneumonia\versions\2"
    r"\chest_xray"
)

splits = ["train", "val", "test"]
categories = ["NORMAL", "PNEUMONIA"]

corrupted_images = []
total_checked = 0

# Check every image
for split in splits:
    for category in categories:

        folder = dataset_path / split / category

        for image_path in folder.glob("*.jpeg"):

            total_checked += 1

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except (OSError, ValueError) as error:
                corrupted_images.append(str(image_path))
                print("Corrupted image:", image_path.name)
                print("Error:", error)

# Display results
print("\n===== DATASET QUALITY REPORT =====")
print("Total images checked:", total_checked)
print("Corrupted images:", len(corrupted_images))


# ------------------------------------------------------------
# 2. EXACT DUPLICATE DETECTION USING SHA-256
# ------------------------------------------------------------

import hashlib

# Store previously observed hashes and their file locations
hash_records = {}

# Store detected duplicate pairs
duplicate_pairs = []

for split in splits:
    for category in categories:

        folder = dataset_path / split / category

        for image_path in folder.glob("*.jpeg"):

            # Calculate the file's SHA-256 hash
            with open(image_path, "rb") as file:
                file_hash = hashlib.file_digest(
                    file, "sha256"
                ).hexdigest()

            # Check whether this hash has appeared before
            if file_hash in hash_records:

                original_path, original_split = hash_records[file_hash]

                duplicate_pairs.append(
                    (original_path, image_path)
                )

                print("\nDUPLICATE DETECTED:")
                print("Original:", original_path)
                print("Duplicate:", image_path)

                if split != original_split:
                    print("WARNING: Duplicate across dataset splits!")

            else:
                hash_records[file_hash] = (image_path, split)

print("\n===== EXACT DUPLICATE REPORT =====")
print("Unique file hashes:", len(hash_records))
print("Duplicate files:", len(duplicate_pairs))

# ------------------------------------------------------------
# 3. IDENTIFY DUPLICATES ACROSS DATASET SPLITS
# ------------------------------------------------------------

from collections import defaultdict

# Group all image locations by their SHA-256 hash
hash_locations = defaultdict(list)

for split in splits:
    for category in categories:

        folder = dataset_path / split / category

        for image_path in folder.glob("*.jpeg"):

            with open(image_path, "rb") as file:
                file_hash = hashlib.file_digest(
                    file, "sha256"
                ).hexdigest()

            hash_locations[file_hash].append(
                (split, category, image_path.name)
            )

# Find hash groups appearing in multiple dataset splits
cross_split_duplicates = {}

for file_hash, locations in hash_locations.items():

    involved_splits = {item[0] for item in locations}

    if len(involved_splits) > 1:
        cross_split_duplicates[file_hash] = locations

print("\n===== CROSS-SPLIT DUPLICATE REPORT =====")
print("Duplicate groups across splits:",
      len(cross_split_duplicates))

for file_hash, locations in cross_split_duplicates.items():

    print("\nMatching files:")

    for split, category, filename in locations:
        print(f"  {split}/{category}/{filename}")

# ------------------------------------------------------------
# 4. COUNT DUPLICATE FILES WITHIN EACH DATASET SPLIT
# ------------------------------------------------------------

print("\n===== WITHIN-SPLIT DUPLICATE REPORT =====")

for split in splits:

    duplicate_count = 0

    for locations in hash_locations.values():

        # Count files belonging to the current split
        count = sum(1 for item in locations if item[0] == split)

        # Count copies beyond the first
        if count > 1:
            duplicate_count += count - 1

    print(f"{split.upper()}: {duplicate_count} duplicate files")