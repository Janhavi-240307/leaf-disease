"""Inference module integrating OpenCV image preprocessing with the trained CNN model."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    MODEL_PATH,
)
from src.disease_info import DISEASE_DETAILS
from src.preprocessor import (
    get_opencv_pipeline_views,
    load_image_from_bytes,
    load_image_from_path,
    preprocess_for_inference,
)


class PotatoDiseasePredictor:
    """Predictor class encapsulating the trained CNN and OpenCV pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the trained Keras model if it exists on disk."""
        if self.model_path.exists():
            print(f"[INFO] Loading inference model from {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
        else:
            self.model = None

    def is_model_ready(self) -> bool:
        """Checks if a trained model is loaded and ready for inference."""
        return self.model is not None

    def predict(
        self, image_input: Union[str, Path, bytes, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Runs complete OpenCV analysis and CNN inference on an input image.
        
        Args:
            image_input: File path, raw bytes from upload/webcam, or BGR ndarray.
            
        Returns:
            Dictionary containing predicted class, confidence, probabilities,
            OpenCV pipeline views, and agronomic disease guidance.
        """
        # 1. Parse input image into BGR numpy array
        if isinstance(image_input, (str, Path)):
            image_bgr = load_image_from_path(str(image_input))
        elif isinstance(image_input, (bytes, bytearray)):
            image_bgr = load_image_from_bytes(image_input)
        elif isinstance(image_input, np.ndarray):
            image_bgr = image_input
        else:
            raise TypeError("Unsupported image input type.")

        # 2. Run OpenCV preprocessing pipeline
        pipeline_views = get_opencv_pipeline_views(image_bgr)

        # 3. Handle case where model is not yet trained
        if not self.is_model_ready():
            return {
                "model_ready": False,
                "views": pipeline_views,
                "message": (
                    "Model weights not found. Please train the model using "
                    "'python src/train.py' before running classification."
                ),
            }

        # 4. Prepare input tensor and predict
        input_tensor = preprocess_for_inference(image_bgr)
        predictions = self.model.predict(input_tensor, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = float(predictions[top_idx])

        # Map probabilities to class names
        prob_dict = {
            CLASS_NAMES[i]: float(prob) for i, prob in enumerate(predictions)
        }

        # 5. Retrieve disease remedies and recommendations
        info = DISEASE_DETAILS.get(predicted_class, {})

        return {
            "model_ready": True,
            "predicted_class": predicted_class,
            "display_name": DISPLAY_NAMES.get(predicted_class, predicted_class),
            "confidence": confidence,
            "probabilities": prob_dict,
            "disease_info": info,
            "views": pipeline_views,
        }

