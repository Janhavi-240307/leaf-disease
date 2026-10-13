"""OpenCV image preprocessing pipeline for leaf segmentation, lesion enhancement, and inspection."""

from typing import Dict, Tuple, Optional
import cv2
import numpy as np

from src.config import IMG_HEIGHT, IMG_WIDTH


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes into a BGR OpenCV NumPy array."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode image from provided bytes.")
    return image_bgr


def load_image_from_path(image_path: str) -> np.ndarray:
    """Loads an image from file path into a BGR OpenCV array."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    return image_bgr


def segment_leaf(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments the leaf and disease spots from background using HSV color thresholding.
    
    Returns:
        (segmented_bgr, binary_mask)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Range 1: Healthy plant green foliage
    lower_green = np.array([25, 30, 30], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Range 2: Necrotic, chlorotic, or dry blight spots (yellowish-brown to dark brown)
    lower_brown = np.array([8, 40, 20], dtype=np.uint8)
    upper_brown = np.array([28, 255, 220], dtype=np.uint8)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combine masks to capture both green leaf tissue and lesion areas
    combined_mask = cv2.bitwise_or(mask_green, mask_brown)

    # Morphological noise cleanup (open to remove specks, close to seal gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply mask over the original image
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=cleaned_mask)

    return segmented, cleaned_mask


def enhance_lesions(image_bgr: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
    to bring out subtle fungal spot textures and leaf surface lesions.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def detect_leaf_contours(
    image_bgr: np.ndarray, binary_mask: np.ndarray
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Finds the main leaf contour and draws a bounding box on a copy of the image.
    
    Returns:
        (annotated_bgr, bounding_box_tuple: (x, y, w, h) or None)
    """
    annotated = image_bgr.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return annotated, None

    # Identify the largest contour corresponding to the leaf body
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500:
        return annotated, None

    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(annotated, [largest_contour], -1, (0, 180, 255), 1)

    return annotated, (x, y, w, h)


def get_opencv_pipeline_views(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Executes the full OpenCV preprocessing pipeline and returns RGB arrays
    ready for Streamlit display.
    """
    segmented_bgr, mask = segment_leaf(image_bgr)
    enhanced_bgr = enhance_lesions(image_bgr)
    annotated_bgr, bbox = detect_leaf_contours(image_bgr, mask)

    # Convert all views to RGB for Streamlit/Matplotlib
    return {
        "original_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "mask": mask,
        "segmented_rgb": cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB),
        "enhanced_rgb": cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB),
        "contour_rgb": cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
        "bbox": bbox,
    }


def preprocess_for_inference(
    image_bgr: np.ndarray, target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
) -> np.ndarray:
    """
    Resizes and prepares image array for TensorFlow/Keras MobileNetV2 input.
    Input range is normalized appropriately for model input [0, 1] or MobileNetV2 scaling.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    expanded = np.expand_dims(resized, axis=0).astype(np.float32)
    return expanded

