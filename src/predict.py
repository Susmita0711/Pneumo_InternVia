"""predict.py
Utility to load a trained model and run inference on a single pre‑processed image.

Functions
    load_model(model_name: str) -> tf.keras.Model
    predict_image(image_path: str, model_name: str = 'cnn') -> Tuple[str, float]
"""

import os
import tensorflow as tf
from typing import Tuple
from .preprocessing import preprocess_path

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def _checkpoint_path(model_name: str) -> str:
    if model_name == "cnn":
        return os.path.join(MODEL_DIR, "best_cnn.h5")
    elif model_name == "mobilenet":
        return os.path.join(MODEL_DIR, "best_mobilenetv2.h5")
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'. Use 'cnn' or 'mobilenet'.")

def load_model(model_name: str = "cnn") -> tf.keras.Model:
    path = _checkpoint_path(model_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {path}. Train the model first or place the .h5 file there."
        )
    return tf.keras.models.load_model(path)

def predict_image(image_path: str, model_name: str = "cnn") -> Tuple[str, float]:
    """Return the predicted class label and confidence score.

    Args:
        image_path: Path to the X‑ray image file.
        model_name: Which trained checkpoint to use ("cnn" or "mobilenet").

    Returns:
        (label, confidence) where label is "NORMAL" or "PNEUMONIA"
        and confidence is a float in [0, 1] (probability of pneumonia).
    """
    model = load_model(model_name)
    img = preprocess_path(image_path)  # shape (1, 224, 224, 3)
    prob = float(model.predict(img, verbose=0).flatten()[0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    return label, prob

