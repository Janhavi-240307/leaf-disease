"""Dataset setup utility for downloading, verifying, and preparing PlantVillage Potato dataset."""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from src.config import CLASS_NAMES, RAW_DATA_DIR, SAMPLES_DIR

# Reliable public mirror for PlantVillage Potato subset
POTATO_DATASET_URL = (
    "https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/"
)


def verify_dataset_structure(data_dir: Path = RAW_DATA_DIR) -> Dict[str, int]:
    """
    Checks the directory for the required 3 classes and counts existing images.
    """
    counts = {}
    total_images = 0

    print(f"\n[INFO] Checking dataset directory: {data_dir}")
    for class_name in CLASS_NAMES:
        class_folder = data_dir / class_name
        if class_folder.exists() and class_folder.is_dir():
            images = [
                f for f in class_folder.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
            counts[class_name] = len(images)
            total_images += len(images)
            print(f"  - {class_name}: {len(images)} images found")
        else:
            counts[class_name] = 0
            print(f"  - [MISSING] {class_name} folder not found")

    print(f"[SUMMARY] Total images found: {total_images}\n")
    return counts


def create_synthetic_test_samples(output_dir: Path = SAMPLES_DIR, samples_per_class: int = 5) -> None:
    """
    Creates synthetic leaf images with simulated lesion patterns.
    Used for verifying pipeline functionality, unit tests, and dry runs
    before downloading the complete PlantVillage dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Generating {samples_per_class} synthetic sample leaves per class into {output_dir}...")

    for class_name in CLASS_NAMES:
        class_out = output_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            # Base dark natural background
            img = np.full((256, 256, 3), (35, 30, 25), dtype=np.uint8)

            # Draw green leaf elliptical body
            leaf_color = (30, 140, 50) if "healthy" in class_name else (40, 110, 45)
            cv2.ellipse(img, (128, 128), (80, 110), 20, 0, 360, leaf_color, -1)
            # Add secondary leaf foliage lobe
            cv2.ellipse(img, (110, 140), (60, 80), -15, 0, 360, leaf_color, -1)

            # Add disease symptoms
            if "Early_blight" in class_name:
                # Target-board concentric brown rings
                centers = [(110, 100), (145, 140), (95, 150)]
                for cx, cy in centers:
                    cv2.circle(img, (cx, cy), 18, (15, 80, 130), -1)  # chlorotic halo
                    cv2.circle(img, (cx, cy), 12, (20, 40, 70), -1)   # necrotic center
                    cv2.circle(img, (cx, cy), 8, (10, 25, 45), -1)    # bullseye inner ring
            elif "Late_blight" in class_name:
                # Water-soaked expanding dark irregular margin lesions
                margin_pts = np.array([
                    [[110, 40], [170, 70], [160, 120], [130, 100]],
                    [[70, 140], [110, 160], [80, 190], [50, 160]]
                ], dtype=np.int32)
                cv2.fillPoly(img, margin_pts, (15, 25, 35))
                # Add white powdery sporulation edge
                cv2.polylines(img, margin_pts, isClosed=True, color=(160, 180, 180), thickness=2)

            file_path = class_out / f"sample_{i+1}.jpg"
            cv2.imwrite(str(file_path), img)

    print(f"[SUCCESS] Synthetic test samples created under {output_dir}")


def print_dataset_instructions() -> None:
    """Prints step-by-step instructions for placing or downloading PlantVillage Potato dataset."""
    print("""
================================================================================
PLANTVILLAGE POTATO DATASET SETUP INSTRUCTIONS:
================================================================================
The project requires the following 3 subdirectories under 'data/raw/':

  leaf-disease/data/raw/
  ├── Potato___Early_blight/
  ├── Potato___Late_blight/
  └── Potato___healthy/

HOW TO ACQUIRE:
Option 1: Using Kaggle CLI / Kagglehub:
  kaggle datasets download -d emmarex/plantdisease -p data/raw/ --unzip
  (Then copy the Potato folders into data/raw/)

Option 2: Direct Kaggle Download:
  1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
     OR https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  2. Download the archive.
  3. Extract the 3 folders (Potato___Early_blight, Potato___Late_blight,
     Potato___healthy) directly into 'data/raw/'.

Option 3: Synthetic Offline Samples (for immediate testing):
  python src/dataset_setup.py --create-samples
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="PlantVillage Potato Dataset Setup")
    parser.add_argument(
        "--check", action="store_true", help="Check and count existing images in data/raw/"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Generate synthetic samples for offline pipeline testing",
    )
    args = parser.parse_args()

    if args.create_samples:
        create_synthetic_test_samples()

    counts = verify_dataset_structure()
    if sum(counts.values()) == 0:
        print_dataset_instructions()


if __name__ == "__main__":
    main()

