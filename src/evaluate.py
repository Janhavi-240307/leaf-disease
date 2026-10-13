"""Model evaluation module: generates confusion matrix and classification reports."""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    MODEL_PATH,
    MODELS_DIR,
    RAW_DATA_DIR,
    SAMPLES_DIR,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)


def evaluate_model(dataset_path: Path = RAW_DATA_DIR, model_path: Path = MODEL_PATH):
    """Loads trained model, predicts validation set, and generates metrics."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python src/train.py' before evaluating."
        )

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print(f"[INFO] Loading evaluation data from: {dataset_path}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        shuffle=False,
    )

    class_names = val_ds.class_names
    y_true = []
    y_pred = []

    print("[INFO] Computing predictions on validation dataset...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Classification Report
    label_indices = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=label_indices, target_names=class_names, digits=4
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    report_file = MODELS_DIR / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"[INFO] Classification report saved to: {report_file}")

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[name.replace("Potato___", "") for name in class_names],
        yticklabels=[name.replace("Potato___", "") for name in class_names],
    )
    plt.title("Confusion Matrix - Potato Leaf Disease", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {CONFUSION_MATRIX_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Potato Leaf Disease Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Path to evaluation dataset directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to saved model .keras file",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Evaluate on synthetic sample dataset",
    )
    args = parser.parse_args()

    data_path = SAMPLES_DIR if args.use_samples else Path(args.data_dir)
    evaluate_model(dataset_path=data_path, model_path=Path(args.model_path))


if __name__ == "__main__":
    main()

