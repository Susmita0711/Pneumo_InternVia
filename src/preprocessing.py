"""preprocessing.py
Utility functions for image preprocessing used in training and inference.
"""

import cv2
import numpy as np
from typing import Tuple

TARGET_SIZE = (224, 224)

def load_image(path: str, to_rgb: bool = True) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Args:
        path: Path to the image file.
        to_rgb: Whether to convert BGR (OpenCV default) to RGB.

    Returns:
        Image as a NumPy array.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image at {path}")
    if to_rgb and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif not to_rgb and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def preprocess_image(img: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize, normalize and expand dimensions for a single image.

    The function expects an image already loaded as a NumPy array. It will:
        * Resize to ``target_size`` keeping aspect ratio via simple interpolation.
        * Scale pixel values to the range [0, 1].
        * Ensure the image has 3 channels (RGB). If a grayscale image is provided,
          it is duplicated across the three channels.
        * Add a batch dimension (``shape`` becomes ``(1, h, w, 3)``).
    """
    # Resize
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Ensure 3 channels
    if len(img_resized.shape) == 2:  # grayscale
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4:
        # Drop alpha channel if present
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)

    # Normalize to [0,1]
    img_norm = img_resized.astype(np.float32) / 255.0

    # Expand batch dimension
    return np.expand_dims(img_norm, axis=0)

def preprocess_path(path: str) -> np.ndarray:
    """Convenience wrapper that loads and preprocesses an image from a file path."""
    img = load_image(path, to_rgb=True)
    return preprocess_image(img)

