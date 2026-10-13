"""Centralized configuration for dataset paths, training parameters, and classes."""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"

# Model Artifacts
MODEL_PATH = MODELS_DIR / "potato_disease_model.keras"
HISTORY_PLOT_PATH = MODELS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Dataset Specs (PlantVillage Potato subset)
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

DISPLAY_NAMES = {
    "Potato___Early_blight": "Potato Early Blight (Alternaria solani)",
    "Potato___Late_blight": "Potato Late Blight (Phytophthora infestans)",
    "Potato___healthy": "Healthy Potato Leaf",
}

# Image & Training Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Ensure key directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

