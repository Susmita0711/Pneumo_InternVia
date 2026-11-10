"""gradcam.py
Utility to generate Grad‑CAM heatmaps for a given model and image.

Functions:
    make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_overlay(img_path, heatmap, output_path, alpha=0.4)
"""
import numpy as np
import tensorflow as tf
import cv2
from .preprocessing import preprocess_path
import os

def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str) -> np.ndarray:
    """Generate a Grad‑CAM heatmap.

    Args:
        img_array: Pre‑processed image of shape (1, h, w, 3).
        model: Keras model used for prediction.
        last_conv_layer_name: Name of the final convolutional layer.

    Returns:
        Heatmap array of shape (h, w) with values in [0, 1].
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    # Compute guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_overlay(img_path: str, heatmap: np.ndarray, output_path: str, alpha: float = 0.4) -> None:
    """Overlay heatmap on the original image and save.

    Args:
        img_path: Original image file.
        heatmap: Heatmap from ``make_gradcam_heatmap``.
        output_path: Where to save the overlay image.
        alpha: Blend factor for the heatmap.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(output_path, overlay)

def generate_gradcam(image_path: str, model: tf.keras.Model, last_conv_layer_name: str, output_dir: str) -> str:
    """Convenient wrapper that returns the path to the saved overlay image.
    """
    os.makedirs(output_dir, exist_ok=True)
    preprocessed = preprocess_path(image_path)  # uses src.preprocessing
    heatmap = make_gradcam_heatmap(preprocessed, model, last_conv_layer_name)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_gradcam.png")
    save_and_overlay(image_path, heatmap, out_path)
    return out_path

