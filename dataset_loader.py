"""Dataset Loader and Validator for PlantVillage Potato Leaf Disease dataset.
Handles automatic downloading, archive extraction, corrupt image validation with OpenCV,
and generation of optimized TensorFlow train/validation data pipelines.
"""

import os
import sys
import shutil
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple, Dict, List, Optional

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)

# Public reliable mirrors for PlantVillage Potato dataset
DATASET_MIRRORS = [
    # GitHub archive containing PlantVillage potato classes
    {
        "url": "https://github.com/ardamavi/Plant-Village-Dataset/archive/refs/heads/master.zip",
        "type": "zip",
        "potato_path_pattern": "Plant-Village-Dataset-master"
    },
    # Secondary mirror: direct tar/zip mirror of PlantVillage potato subsets
    {
        "url": "https://storage.googleapis.com/plantvillage-subset/potato_dataset.zip",
        "type": "zip",
        "potato_path_pattern": "potato"
    }
]


def download_with_progress(url: str, destination: Path) -> bool:
    """Downloads a remote file with console progress indicator."""
    print(f"[INFO] Downloading from: {url}")
    print(f"[INFO] Saving to: {destination}")

    try:
        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)")
            else:
                sys.stdout.write(f"\r  Downloaded: {downloaded / (1024 * 1024):.1f}MB")
            sys.stdout.flush()

        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(destination, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', -1))
            block_size = 1024 * 64
            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
                else:
                    sys.stdout.write(f"\r  Downloaded: {downloaded / (1024*1024):.1f}MB")
                sys.stdout.flush()
        print("\n[SUCCESS] Download completed successfully.")
        return True
    except Exception as e:
        print(f"\n[WARNING] Download failed from {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False


def validate_and_clean_images(data_dir: Path) -> Dict[str, int]:
    """
    Scans every image using OpenCV to verify file integrity.
    Removes corrupt, unreadable, or 0-byte images.
    Returns dictionary with valid image counts per class.
    """
    print(f"\n[INFO] Validating image integrity with OpenCV in: {data_dir}")
    counts = {}
    total_valid = 0
    total_corrupt = 0

    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            counts[class_name] = 0
            continue

        valid_count = 0
        image_files = [f for f in class_dir.iterdir() if f.is_file()]

        for img_path in image_files:
            # Check extension
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Check if OpenCV can decode the image
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0 or len(img.shape) != 3:
                    print(f"  [CORRUPT] Removing unreadable image: {img_path.name}")
                    img_path.unlink()
                    total_corrupt += 1
                else:
                    valid_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to read {img_path.name}: {e}. Removing.")
                try:
                    img_path.unlink()
                except Exception:
                    pass
                total_corrupt += 1

        counts[class_name] = valid_count
        total_valid += valid_count
        display_label = DISPLAY_NAMES.get(class_name, class_name)
        print(f"  [OK] {display_label}: {valid_count} verified images")

    print(f"[SUMMARY] Total verified images: {total_valid} | Corrupted removed: {total_corrupt}\n")
    return counts


def ensure_dataset(data_dir: Path = RAW_DATA_DIR) -> Dict[str, int]:
    """
    Ensures that the real PlantVillage Potato dataset is downloaded,
    structured, and validated.
    """
    counts = validate_and_clean_images(data_dir)
    total_existing = sum(counts.values())

    if total_existing > 50:
        print(f"[INFO] Dataset already present with {total_existing} valid images.")
        return counts

    print("[INFO] Dataset not found or incomplete. Attempting automatic download...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: Git sparse-checkout from official PlantVillage repo (fast & reliable)
    try:
        import subprocess
        print("[INFO] Trying Git sparse-checkout from official spMohanty/PlantVillage-Dataset...")
        temp_repo = data_dir.parent / "temp_git_plantvillage"
        if temp_repo.exists():
            shutil.rmtree(temp_repo, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
             "https://github.com/spMohanty/PlantVillage-Dataset.git", str(temp_repo)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=str(temp_repo), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(
            ["git", "sparse-checkout", "set",
             "raw/color/Potato___Early_blight",
             "raw/color/Potato___Late_blight",
             "raw/color/Potato___healthy"],
            cwd=str(temp_repo), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(["git", "checkout"], cwd=str(temp_repo), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        src_base = temp_repo / "raw" / "color"
        for cls in CLASS_NAMES:
            src_cls = src_base / cls
            dest_cls = data_dir / cls
            dest_cls.mkdir(parents=True, exist_ok=True)
            if src_cls.exists():
                for f in src_cls.glob("*.*"):
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        shutil.copy2(f, dest_cls / f.name)

        shutil.rmtree(temp_repo, ignore_errors=True)
        counts = validate_and_clean_images(data_dir)
        if sum(counts.values()) > 50:
            print("[SUCCESS] PlantVillage Potato dataset acquired via Git sparse-checkout!")
            return counts
    except Exception as ge:
        print(f"[WARNING] Git sparse-checkout attempt encountered error: {ge}. Falling back to mirrors...")
        if (data_dir.parent / "temp_git_plantvillage").exists():
            shutil.rmtree(data_dir.parent / "temp_git_plantvillage", ignore_errors=True)

    # Method 2: HTTP Zip mirrors fallback
    archive_path = data_dir.parent / "potato_dataset_download.zip"
    downloaded = False
    for mirror in DATASET_MIRRORS:
        if download_with_progress(mirror["url"], archive_path):
            downloaded = True
            try:
                print(f"[INFO] Extracting archive: {archive_path}")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # Extract to temporary extraction directory
                    temp_extract = data_dir.parent / "temp_extract"
                    zip_ref.extractall(temp_extract)

                    # Search for potato folders inside extracted content
                    for root, dirs, _ in os.walk(temp_extract):
                        for d in dirs:
                            lower_d = d.lower()
                            matched_class = None
                            if "early_blight" in lower_d or "early blight" in lower_d:
                                matched_class = "Potato___Early_blight"
                            elif "late_blight" in lower_d or "late blight" in lower_d:
                                matched_class = "Potato___Late_blight"
                            elif "healthy" in lower_d and "potato" in lower_d:
                                matched_class = "Potato___healthy"

                            if matched_class:
                                src_folder = Path(root) / d
                                dest_folder = data_dir / matched_class
                                dest_folder.mkdir(parents=True, exist_ok=True)
                                for f in src_folder.glob("*.*"):
                                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                                        shutil.copy2(f, dest_folder / f.name)

                    # Cleanup temp extract
                    shutil.rmtree(temp_extract, ignore_errors=True)
                break
            except Exception as e:
                print(f"[ERROR] Failed to extract archive: {e}")
            finally:
                if archive_path.exists():
                    archive_path.unlink()

    counts = validate_and_clean_images(data_dir)
    return counts


def create_data_pipelines(
    data_dir: Path = RAW_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    image_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
    seed: int = RANDOM_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Creates high-performance, prefetched TensorFlow training and validation data pipelines.
    Includes data augmentation (random flip, rotation, zoom) to prevent overfitting.
    """
    print(f"[INFO] Building TensorFlow data pipelines from: {data_dir}")

    # Build base train dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        labels="inferred",
        label_mode="int",
    )

    # Build base validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        labels="inferred",
        label_mode="int",
    )

    class_names = train_ds.class_names
    print(f"[INFO] Target classes identified: {class_names}")

    # Data Augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation_pipeline")

    autotune = tf.data.AUTOTUNE

    # Apply augmentation to training dataset only, then cache and prefetch
    train_ds = (
        train_ds
        .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)
        .cache()
        .prefetch(buffer_size=autotune)
    )

    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


if __name__ == "__main__":
    ensure_dataset()

