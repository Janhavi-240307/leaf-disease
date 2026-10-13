"""Root-level Training Script for Potato Leaf Disease Classification.
Trains MobileNetV2 Transfer Learning model, evaluates performance,
and saves model weights, class indices, and evaluation plots.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Ensure local imports work cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    RAW_DATA_DIR,
    MODELS_DIR,
    MODEL_PATH,
    HISTORY_PLOT_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)
from src.model import build_transfer_model
from dataset_loader import ensure_dataset, create_data_pipelines


def plot_and_save_curves(history, save_path: Path = HISTORY_PLOT_PATH) -> None:
    """Plots training and validation accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o", color="#2e7d32")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="s", color="#1565c0")
    plt.title("Model Accuracy Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o", color="#c62828")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="s", color="#ef6c00")
    plt.title("Model Loss Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved training progress curves to: {save_path}")


def evaluate_and_save_metrics(model: tf.keras.Model, val_ds: tf.data.Dataset, class_names: list) -> dict:
    """Computes confusion matrix, classification report, and saves artifacts."""
    print("\n[INFO] Evaluating trained model on validation set...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification Report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, digits=4, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )

    print("\n" + "=" * 65)
    print("VALIDATION CLASSIFICATION REPORT")
    print("=" * 65)
    print(report_text)

    with open(MODELS_DIR / "classification_report.txt", "w") as f:
        f.write(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    clean_labels = [name.replace("Potato___", "").replace("_", " ").title() for name in class_names]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
    )
    plt.title("Confusion Matrix - Validation Set", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Condition")
    plt.ylabel("True Condition")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix heatmap to: {CONFUSION_MATRIX_PATH}")

    overall_accuracy = float(np.mean(y_true == y_pred))
    return {
        "accuracy": overall_accuracy,
        "classification_report": report_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Potato Leaf Disease CNN Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ensure dataset exists and validate images
    counts = ensure_dataset(RAW_DATA_DIR)
    total_imgs = sum(counts.values())
    if total_imgs == 0:
        print("[ERROR] No valid images available for training. Exiting.")
        sys.exit(1)

    # 2. Build TensorFlow data pipelines
    train_ds, val_ds, class_names = create_data_pipelines(
        data_dir=RAW_DATA_DIR,
        batch_size=args.batch_size,
    )

    # Save class mapping index for prediction
    class_indices = {i: name for i, name in enumerate(class_names)}
    with open(MODELS_DIR / "class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=4)
    print(f"[INFO] Saved class indices mapping to {MODELS_DIR / 'class_indices.json'}")

    # 3. Build Transfer Learning Model
    print(f"\n[INFO] Initializing MobileNetV2 Transfer Learning Architecture...")
    model = build_transfer_model(
        num_classes=len(class_names),
        learning_rate=args.lr,
    )

    # 4. Configure Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Fit Model
    print(f"\n[INFO] Starting model training for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Ensure best model is saved
    model.save(str(MODEL_PATH))
    print(f"\n[SUCCESS] Final model saved to: {MODEL_PATH}")

    # 6. Plot Curves
    plot_and_save_curves(history)

    # 7. Evaluate and Save Metrics
    eval_metrics = evaluate_and_save_metrics(model, val_ds, class_names)

    # Save comprehensive metrics json
    final_metrics = {
        "validation_accuracy": eval_metrics["accuracy"],
        "epochs_trained": len(history.history["accuracy"]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "class_counts": counts,
        "classes": class_names,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"[SUCCESS] All metrics successfully saved to {METRICS_PATH}")
    print(f"  → Final Validation Accuracy: {eval_metrics['accuracy'] * 100:.2f}%\n")


if __name__ == "__main__":
    main()

