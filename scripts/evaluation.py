"""
Evaluation Script — India AQI Classifier
=========================================
Runs inside SageMaker SKLearnProcessor.

Outputs evaluation.json consumed by:
  • ConditionStep  — metrics.accuracy.value ≥ threshold
  • ModelMetrics   — attached to registered model package

Multi-class metrics computed:
  accuracy, weighted F1, macro F1, weighted AUC (OvR),
  per-class precision/recall/F1,
  confusion matrix
"""

import json
import logging
import os
import tarfile
import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR  = "/opt/ml/processing/model"
TEST_DIR   = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"

AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def extract_model(model_dir: str) -> str:
    tar = os.path.join(model_dir, "model.tar.gz")
    if not os.path.exists(tar):
        return model_dir
    out = os.path.join(model_dir, "extracted")
    os.makedirs(out, exist_ok=True)
    with tarfile.open(tar, "r:gz") as t:
        t.extractall(out)
    return out


def load_test(test_dir: str):
    files = glob.glob(os.path.join(test_dir, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.iloc[:, 1:].values, df.iloc[:, 0].values.astype(int)


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc    = accuracy_score(y, y_pred)
    f1_w   = f1_score(y, y_pred, average="weighted", zero_division=0)
    f1_mac = f1_score(y, y_pred, average="macro",    zero_division=0)
    prec   = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y, y_pred, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(y, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    # Per-class metrics
    n_classes = model.n_classes_
    per_class = {}
    for cls_idx in range(n_classes):
        label = AQI_LABELS[cls_idx] if cls_idx < len(AQI_LABELS) else str(cls_idx)
        mask  = (y == cls_idx)
        if mask.sum() > 0:
            per_class[label] = {
                "precision": round(float(precision_score(y == cls_idx, y_pred == cls_idx, zero_division=0)), 4),
                "recall":    round(float(recall_score(y == cls_idx, y_pred == cls_idx, zero_division=0)), 4),
                "f1":        round(float(f1_score(y == cls_idx, y_pred == cls_idx, zero_division=0)), 4),
                "support":   int(mask.sum()),
            }

    report = {
        "metrics": {
            "accuracy":  {"value": round(acc,    4)},
            "f1":        {"value": round(f1_w,   4)},
            "f1_macro":  {"value": round(f1_mac, 4)},
            "precision": {"value": round(prec,   4)},
            "recall":    {"value": round(rec,     4)},
        },
        "per_class_metrics":  per_class,
        "confusion_matrix":   confusion_matrix(y, y_pred).tolist(),
        "num_test_samples":   int(len(y)),
        "class_labels":       AQI_LABELS,
    }

    if auc is not None:
        report["metrics"]["auc"] = {"value": round(auc, 4)}

    logger.info("Evaluation metrics:\n%s",
                json.dumps(report["metrics"], indent=2))
    logger.info("Per-class metrics:\n%s",
                json.dumps(per_class, indent=2))
    logger.info("Classification report:\n%s",
                classification_report(y, y_pred, target_names=AQI_LABELS[:n_classes]))
    return report


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extracted = extract_model(MODEL_DIR)
    model = joblib.load(os.path.join(extracted, "model.joblib"))
    X_test, y_test = load_test(TEST_DIR)

    report = evaluate(model, X_test, y_test)

    out_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Evaluation report → %s", out_path)


if __name__ == "__main__":
    main()