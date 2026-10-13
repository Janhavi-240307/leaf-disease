"""Root-level Prediction and Inference Utility.
Integrates OpenCV image preprocessing with the trained TensorFlow CNN model.
Can be executed via CLI or imported into other modules.
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
from typing import Dict, Any, Union, Optional
import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    MODEL_PATH,
    MODELS_DIR,
    SAMPLES_DIR,
    RAW_DATA_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
)
from src.disease_info import DISEASE_DETAILS
from src.preprocessor import (
    get_opencv_pipeline_views,
    load_image_from_bytes,
    load_image_from_path,
    preprocess_for_inference,
)


class DiseaseClassifier:
    """End-to-end classifier integrating OpenCV and trained Keras model."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self.class_mapping = self._load_class_indices()
        self._load_model()

    def _load_class_indices(self) -> Dict[int, str]:
        indices_path = MODELS_DIR / "class_indices.json"
        if indices_path.exists():
            try:
                with open(indices_path, "r") as f:
                    data = json.load(f)
                    return {int(k): v for k, v in data.items()}
            except Exception:
                pass
        return {i: name for i, name in enumerate(CLASS_NAMES)}

    def _load_model(self) -> None:
        if self.model_path.exists():
            print(f"[INFO] Loading trained CNN weights from: {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
        else:
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image_input: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """Runs OpenCV preprocessing, lesion enhancement, and CNN inference."""
        if isinstance(image_input, (str, Path)):
            img_bgr = load_image_from_path(str(image_input))
        elif isinstance(image_input, (bytes, bytearray)):
            img_bgr = load_image_from_bytes(image_input)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise TypeError("Unsupported image input format.")

        views = get_opencv_pipeline_views(img_bgr)

        if not self.is_ready():
            return {
                "success": False,
                "message": f"Model file not found at '{self.model_path}'. Please train the model with 'python train.py' first.",
                "views": views,
            }

        input_tensor = preprocess_for_inference(img_bgr, (IMG_HEIGHT, IMG_WIDTH))
        preds = self.model.predict(input_tensor, verbose=0)[0]

        top_idx = int(np.argmax(preds))
        predicted_class = self.class_mapping.get(top_idx, CLASS_NAMES[top_idx])
        confidence = float(preds[top_idx])

        probabilities = {
            self.class_mapping.get(i, CLASS_NAMES[i]): float(p)
            for i, p in enumerate(preds)
        }

        disease_info = DISEASE_DETAILS.get(predicted_class, {})

        return {
            "success": True,
            "predicted_class": predicted_class,
            "display_name": DISPLAY_NAMES.get(predicted_class, predicted_class),
            "confidence": confidence,
            "probabilities": probabilities,
            "views": views,
            "disease_info": disease_info,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Plant Leaf Disease using OpenCV & CNN")
    parser.add_argument("--image", type=str, help="Path to input potato leaf image file")
    parser.add_argument("--save-vis", type=str, help="Save OpenCV visualization grid to file")
    parser.add_argument("--test", action="store_true", help="Run prediction on any available test image")
    args = parser.parse_args()

    image_path = None
    if args.image:
        image_path = Path(args.image)
    elif args.test:
        candidates = list(RAW_DATA_DIR.glob("**/*.jpg")) + list(SAMPLES_DIR.glob("**/*.jpg"))
        if candidates:
            image_path = candidates[0]
            print(f"[INFO] Using test image candidate: {image_path}")
        else:
            print("[ERROR] No sample or dataset images found. Specify an image with --image.")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    if not image_path.exists():
        print(f"[ERROR] Image not found at {image_path}")
        sys.exit(1)

    classifier = DiseaseClassifier()
    result = classifier.predict(image_path)

    print("\n" + "=" * 60)
    print("DIAGNOSIS RESULTS")
    print("=" * 60)
    if result["success"]:
        print(f"Condition:  {result['display_name']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%\n")
        print("Probability Breakdown:")
        for cls_name, prob in result["probabilities"].items():
            print(f"  - {DISPLAY_NAMES.get(cls_name, cls_name):<45}: {prob * 100:.2f}%")

        if "disease_info" in result and result["disease_info"]:
            info = result["disease_info"]
            print("\nAgronomic Guidance:")
            print(f"Pathogen: {info.get('pathogen', 'N/A')}")
            print(f"Severity: {info.get('severity', 'N/A')}")
            print("\nRecommended Organic Actions:")
            for item in info.get("organic_control", []):
                print(f"  • {item}")
            print("\nChemical Interventions:")
            for item in info.get("chemical_control", []):
                print(f"  • {item}")
    else:
        print(f"Status: {result['message']}")

    if args.save_vis and "views" in result:
        views = result["views"]
        vis_path = Path(args.save_vis)
        # Convert RGB segmented view to BGR for saving
        cv2.imwrite(str(vis_path), cv2.cvtColor(views["segmented_rgb"], cv2.COLOR_RGB2BGR))
        print(f"\n[INFO] Saved preprocessed visualization to: {vis_path}")


if __name__ == "__main__":
    main()

