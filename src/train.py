"""train.py
Train the PneumoCNN (and optionally MobileNetV2) on the Chest X‑Ray dataset.

Usage::
    python src/train.py --epochs 20 --batch-size 32 --model cnn
    python src/train.py --epochs 10 --batch-size 16 --model mobilenet

The script:
    * Sets a deterministic random seed.
    * Loads training and validation datasets via ``src.dataset_loader``.
    * Instantiates the requested model architecture.
    * Configures callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
    * Fits the model and saves the final checkpoint.

If the dataset directory is not found, a clear error message is printed and the script exits.
"""

import argparse
import os
import sys
import random
import numpy as np
import tensorflow as tf

from src.dataset_loader import load_dataset
from src.model import build_cnn, build_mobilenetv2

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PneumoScan models")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet"],
        default="cnn",
        help="Which architecture to train",
    )
    parser.add_argument(
        "--fine-tune-at",
        type=int,
        default=100,
        help="Layer index to start fine‑tuning for MobileNetV2",
    )
    return parser.parse_args()

def ensure_dataset_exists():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.isdir(data_root):
        print(
            "[ERROR] Dataset directory not found.\n"
            "Please download the Chest X‑Ray Pneumonia dataset from Kaggle and place it under the project root as 'data/'.\n"
            f"Expected path: {data_root}\n"
        )
        sys.exit(1)

def main():
    args = parse_args()
    ensure_dataset_exists()

    # Load datasets
    train_ds = load_dataset(split="train", batch_size=args.batch_size, shuffle=True)
    val_ds = load_dataset(split="val", batch_size=args.batch_size, shuffle=False)

    # Model selection
    if args.model == "cnn":
        model = build_cnn()
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_cnn.h5"))
    else:
        model = build_mobilenetv2(fine_tune_at=args.fine_tune_at)
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_mobilenetv2.h5"))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True, monitor="val_loss", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    # Training
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # Save final model (in case EarlyStopping never triggered)
    final_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", f"final_{args.model}.h5"))
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")

    # Optionally persist training curves for the UI
    import pandas as pd
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "training_history.csv")), index=False)

if __name__ == "__main__":
    main()

