"""dataset_loader.py
Utilities for loading the Chest X-ray Pneumonia dataset.
"""
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def get_image_paths_and_labels(split: str = "train") -> Tuple[List[Path], List[int]]:
    """Return image file paths and integer labels for a given split.

    Args:
        split: One of "train", "val", "test".
    Returns:
        A tuple (paths, labels) where labels are 0 for NORMAL and 1 for PNEUMONIA.
    """
    base = DATA_DIR / split
    normal_dir = base / "NORMAL"
    pneumonia_dir = base / "PNEUMONIA"

    normal_paths = list(normal_dir.rglob("*.jpeg")) + list(normal_dir.rglob("*.jpg")) + list(normal_dir.rglob("*.png"))
    pneumonia_paths = list(pneumonia_dir.rglob("*.jpeg")) + list(pneumonia_dir.rglob("*.jpg")) + list(pneumonia_dir.rglob("*.png"))

    paths = normal_paths + pneumonia_paths
    labels = [0] * len(normal_paths) + [1] * len(pneumonia_paths)
    return paths, labels

def load_dataset(split: str = "train", batch_size: int = 32, shuffle: bool = True):
    """Create a TensorFlow Dataset for the specified split.
    The dataset yields (image_tensor, label) pairs.
    """
    from tensorflow.data import Dataset
    paths, labels = get_image_paths_and_labels(split)
    if shuffle:
        combined = list(zip(paths, labels))
        np.random.shuffle(combined)
        paths, labels = zip(*combined)
    # Convert to list for tf.data
    path_ds = Dataset.from_tensor_slices(list(map(str, paths)))
    label_ds = Dataset.from_tensor_slices(list(labels))

    def _load_preprocess(path, label):
        img = preprocess_path(path.numpy().decode())
        return img[0], label

    def _tf_wrapper(path, label):
        img, lbl = tf.py_function(_load_preprocess, [path, label], [tf.float32, tf.int32])
        img.set_shape([224, 224, 3])
        lbl.set_shape([])
        return img, lbl

    import tensorflow as tf
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(_tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

