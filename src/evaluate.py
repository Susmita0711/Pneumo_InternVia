"""evaluate.py
Evaluate a trained PneumoScan model on the untouched test set and generate performance reports.

Usage::
    python src/evaluate.py --model cnn   # evaluates best_cnn.h5
    python src/evaluate.py --model mobilenet

The script:
    * Loads the appropriate checkpoint from ``models/``.
    * Runs inference on the whole test split.
    * Computes accuracy, precision, recall, specificity, F1, ROC‑AUC.
    * Saves a confusion matrix heatmap and ROC curve under ``assets/``.
    * Prints a concise classification report.
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.dataset_loader import load_dataset

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PneumoScan model")
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet"],
        default="cnn",
        help="Which checkpoint to evaluate",
    )
    return parser.parse_args()

def load_checkpoint(model_name: str):
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    if model_name == "cnn":
        ckpt_path = os.path.join(ckpt_dir, "best_cnn.h5")
    else:
        ckpt_path = os.path.join(ckpt_dir, "best_mobilenetv2.h5")
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found at {ckpt_path}. Train the model first.")
        sys.exit(1)
    return tf.keras.models.load_model(ckpt_path)

def compute_specificity(cm: np.ndarray) -> float:
    # cm = [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def main():
    args = parse_args()
    model = load_checkpoint(args.model)

    test_ds = load_dataset(split="test", batch_size=32, shuffle=False)

    y_true = []
    y_pred_prob = []
    for batch_imgs, batch_labels in test_ds:
        probs = model.predict(batch_imgs, verbose=0).flatten()
        y_pred_prob.extend(probs.tolist())
        y_true.extend(batch_labels.numpy().tolist())
    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = compute_specificity(confusion_matrix(y_true, y_pred))
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)

    print("=== Test Set Metrics ===")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall (Sensitivity) : {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1‑Score   : {f1:.4f}")
    print(f"ROC‑AUC    : {auc:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "confusion_matrix.png"))
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "roc_curve.png"))
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Save numeric metrics for UI consumption
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC‑AUC"],
            "Value": [acc, prec, rec, spec, f1, auc],
        }
    )
    metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "test_metrics.csv"))
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved confusion matrix to {cm_path}\nSaved ROC curve to {roc_path}\nSaved numeric metrics to {metrics_path}")

if __name__ == "__main__":
    main()

