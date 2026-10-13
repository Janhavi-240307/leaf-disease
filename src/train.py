"""Training pipeline using TensorFlow/Keras and MobileNetV2 Transfer Learning."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EPOCHS,
    HISTORY_PLOT_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    LEARNING_RATE,
    MODEL_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RAW_DATA_DIR,
    SAMPLES_DIR,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)
from src.model import build_transfer_model, build_custom_cnn


def get_data_augmentation() -> tf.keras.Sequential:
    """Builds a data augmentation pipeline using Keras layers."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def load_datasets(dataset_path: Path):
    """Loads training and validation datasets from a directory structure."""
    print(f"\n[INFO] Loading images from: {dataset_path}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        labels="inferred",
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        labels="inferred",
        label_mode="int",
    )

    class_names = train_ds.class_names
    print(f"[INFO] Discovered classes ({len(class_names)}): {class_names}")

    # Optimize pipeline with caching and prefetching
    augmentation = get_data_augmentation()
    autotune = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=autotune)
        .cache()
        .prefetch(buffer_size=autotune)
    )

    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def plot_history(history, save_path: Path = HISTORY_PLOT_PATH) -> None:
    """Plots and saves accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="s")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="s")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


def train(
    dataset_path: Path = RAW_DATA_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_type: str = "transfer",
) -> None:
    """Main training execution function."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if dataset directory exists and has subdirectories
    if not dataset_path.exists() or not any(dataset_path.iterdir()):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' is empty or does not exist. "
            "Please run 'python src/dataset_setup.py' first."
        )

    train_ds, val_ds, classes = load_datasets(dataset_path)

    # Build model
    if model_type == "custom":
        print("[INFO] Building Custom 4-stage CNN model...")
        model = build_custom_cnn(num_classes=len(classes), learning_rate=LEARNING_RATE)
    else:
        print("[INFO] Building MobileNetV2 Transfer Learning model...")
        model = build_transfer_model(num_classes=len(classes), learning_rate=LEARNING_RATE)

    model.summary()

    # Callbacks
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
            patience=5,
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

    print(f"\n[INFO] Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save final model if not already saved by checkpoint
    model.save(str(MODEL_PATH))
    print(f"[SUCCESS] Model saved successfully to: {MODEL_PATH}")

    # Plot & save history curves
    plot_history(history)

    # Save metrics metadata
    final_metrics = {
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "classes": classes,
        "epochs_trained": len(history.history["accuracy"]),
        "architecture": model_type,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"[INFO] Metrics saved to: {METRICS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train Potato Leaf Disease CNN")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Path to training data root folder",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Training batch size"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["transfer", "custom"],
        default="transfer",
        help="Model architecture: 'transfer' (MobileNetV2) or 'custom' CNN",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Train on samples directory for quick pipeline dry-run",
    )
    args = parser.parse_args()

    data_path = SAMPLES_DIR if args.use_samples else Path(args.data_dir)
    train(
        dataset_path=data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()

