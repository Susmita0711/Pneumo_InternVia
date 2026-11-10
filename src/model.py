"""model.py
Defines the CNN architecture used for PneumoScan and an optional MobileNetV2 transfer‑learning version.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_cnn(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """Construct a modest custom CNN.

    Architecture (illustrative):
        Conv2D(32) → BN → ReLU → MaxPool
        Conv2D(64) → BN → ReLU → MaxPool
        Conv2D(128) → BN → ReLU → MaxPool
        GlobalAveragePooling
        Dense(128) → ReLU → Dropout(0.5)
        Dense(1) → Sigmoid
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="PneumoCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_mobilenetv2(input_shape=(224, 224, 3), fine_tune_at=100) -> tf.keras.Model:
    """MobileNetV2‑based transfer learning model.

    * Loads ImageNet‑pretrained MobileNetV2 (excluding top).
    * Freezes layers up to ``fine_tune_at``.
    * Adds global pooling, dropout and a sigmoid head.
    """
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(base_model.input, outputs, name="PneumoMobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

