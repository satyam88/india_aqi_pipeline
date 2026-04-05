"""
Training Script — India AQI Classifier
=======================================
Runs inside SageMaker SKLearn estimator container.

Task: 6-class AQI category prediction
  0 = Good  1 = Satisfactory  2 = Moderate
  3 = Poor  4 = Very Poor      5 = Severe

Algorithm: RandomForestClassifier
Reasons:
  • Handles -1 sentinel values (missing pollutant sensors) naturally
  • Non-linear PM2.5 / PM10 interaction effects captured without transforms
  • class_weight="balanced" compensates for rare Severe/Good categories
  • Feature importance is directly interpretable (µg/m³ → AQI)
  • Robust to city-level distribution shift (OOB score reflects this)

Expected validation performance:
  Accuracy ~78-82%    (6-class problem, harder than binary)
  Macro F1  ~75-80%
  AUC (OvR)  ~93%
"""

import argparse
import io
import json as _json
import logging
import os
import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRAIN_DIR  = os.environ.get("SM_CHANNEL_TRAIN",     "/opt/ml/input/data/train")
VAL_DIR    = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
MODEL_DIR  = os.environ.get("SM_MODEL_DIR",          "/opt/ml/model")

AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
NUM_COLS   = ["pm25", "pm10", "no2", "so2", "co", "o3"]


def load_split(directory: str):
    files = glob.glob(os.path.join(directory, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    y = df.iloc[:, 0].values.astype(int)   # first column = aqi_category
    X = df.iloc[:, 1:].values
    logger.info("Loaded %d rows × %d features from %s", len(df), X.shape[1], directory)
    return X, y


def train(X, y, params: dict) -> RandomForestClassifier:
    logger.info("Training RandomForest: %s", params)
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"] or None,
        min_samples_split=params["min_samples_split"],
        class_weight=params.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X, y)
    logger.info("OOB score: %.4f", model.oob_score_)
    return model


def evaluate(model, X, y, split: str) -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)

    acc    = accuracy_score(y, preds)
    f1     = f1_score(y, preds, average="weighted")
    f1_mac = f1_score(y, preds, average="macro")

    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    print(f"{split} accuracy: {acc:.4f}")
    print(f"{split} f1: {f1:.4f}")
    print(f"{split} f1_macro: {f1_mac:.4f}")
    print(f"{split} auc: {auc:.4f}")

    logger.info("%s  acc=%.4f  f1_w=%.4f  f1_mac=%.4f  auc=%.4f",
                split, acc, f1, f1_mac, auc)

    n_classes = model.n_classes_
    labels_present = [AQI_LABELS[i] for i in range(n_classes) if i < len(AQI_LABELS)]
    logger.info(
        "Classification report:\n%s",
        classification_report(y, preds, target_names=labels_present[:n_classes]),
    )

    return {"accuracy": acc, "f1": f1, "f1_macro": f1_mac, "auc": auc}


def log_feature_importance(model, n_features: int) -> None:
    imp   = model.feature_importances_
    top_n = min(20, n_features)
    top_idx = np.argsort(imp)[::-1][:top_n]
    logger.info("Top %d feature importances:", top_n)
    for rank, idx in enumerate(top_idx, 1):
        logger.info("  %2d. feature_%d  importance=%.4f", rank, idx, imp[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",      type=int, default=300)
    parser.add_argument("--max-depth",         type=int, default=15, help="0=unlimited")
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--class-weight",      type=str, default="balanced")
    args = parser.parse_args()

    X_train, y_train = load_split(TRAIN_DIR)
    X_val,   y_val   = load_split(VAL_DIR)

    params = {
        "n_estimators":      args.n_estimators,
        "max_depth":         args.max_depth if args.max_depth > 0 else None,
        "min_samples_split": args.min_samples_split,
        "class_weight":      args.class_weight,
    }

    model = train(X_train, y_train, params)
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val,   y_val,   "validation")
    log_feature_importance(model, X_train.shape[1])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
    logger.info("Model saved → %s/model.joblib", MODEL_DIR)

    # ── Copy scaler + columns from training channel so they're packaged
    #    into model.tar.gz alongside model.joblib ──────────────────────────
    for fname in ("scaler.joblib", "columns.json"):
        src = os.path.join(TRAIN_DIR, fname)
        dst = os.path.join(MODEL_DIR, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            logger.info("Copied %s → %s", fname, dst)
        else:
            logger.warning("%s not found in TRAIN_DIR — inference may fail", fname)


# ── SageMaker inference handlers ──────────────────────────────────────────

def model_fn(model_dir: str) -> dict:
    """Load model, scaler and column list from model_dir."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    scaler_path = os.path.join(model_dir, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    if scaler is None:
        logger.warning("scaler.joblib not found — numeric features will NOT be scaled")

    col_path = os.path.join(model_dir, "columns.json")
    if os.path.exists(col_path):
        with open(col_path) as f:
            columns = _json.load(f)
    else:
        columns = None
        logger.warning("columns.json not found — column alignment will be skipped")

    return {"model": model, "scaler": scaler, "columns": columns}


def input_fn(body: str, content_type: str = "application/json") -> pd.DataFrame:
    """
    Accept two payload formats:

    1. High-level (location + datetime, optionally pollutants):
       {
         "location": "Delhi",
         "datetime": "2026-03-10T08:00:00",
         "pm25": 85.2,   # optional — defaults to -1 (missing sentinel)
         "pm10": 120.0,  # optional
         ...
       }

    2. Raw numeric (legacy):
       {"instances": [[f1, f2, ...]]}
       {"inputs":    [[f1, f2, ...]]}
    """
    if content_type == "text/csv":
        return pd.read_csv(io.StringIO(body), header=None)

    if content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")

    data = _json.loads(body)

    # ── Legacy raw numeric payload ─────────────────────────────────────────
    if "instances" in data or "inputs" in data:
        arr = data.get("instances") or data["inputs"]
        return pd.DataFrame(np.array(arr, dtype=np.float32))

    # ── High-level location/datetime payload ──────────────────────────────
    if "location" not in data or "datetime" not in data:
        raise ValueError(
            "Payload must contain either 'instances'/'inputs' (raw features) "
            "or 'location' + 'datetime' keys."
        )

    dt = pd.Timestamp(data["datetime"])

    row = {
        # Pollutant readings (-1 = missing sentinel, safe for RandomForest)
        "pm25": float(data.get("pm25", -1)),
        "pm10": float(data.get("pm10", -1)),
        "no2":  float(data.get("no2",  -1)),
        "so2":  float(data.get("so2",  -1)),
        "co":   float(data.get("co",   -1)),
        "o3":   float(data.get("o3",   -1)),
        # Time features — must match preprocess.py exactly
        "hour_of_day":  dt.hour,
        "month":        dt.month,
        "is_rush_hour": int(dt.hour in [7, 8, 9, 17, 18, 19]),
        "is_weekend":   int(dt.dayofweek >= 5),
        # City one-hot column (get_dummies style: city_<Name>)
        f"city_{data['location']}": 1,
    }

    return pd.DataFrame([row])


def predict_fn(input_df: pd.DataFrame, model_artifacts: dict) -> dict:
    model   = model_artifacts["model"]
    scaler  = model_artifacts["scaler"]
    columns = model_artifacts["columns"]

    # ── Align to training column order ────────────────────────────────────
    if columns is not None:
        input_df = input_df.reindex(columns=columns, fill_value=0)

    # ── Scale numeric columns ─────────────────────────────────────────────
    if scaler is not None:
        num_present = [c for c in NUM_COLS if c in input_df.columns]
        if num_present:
            input_df = input_df.copy()
            input_df[num_present] = scaler.transform(input_df[num_present])

    preds  = model.predict(input_df).tolist()
    labels = [AQI_LABELS[p] if p < len(AQI_LABELS) else str(p) for p in preds]
    proba  = model.predict_proba(input_df).tolist()

    return {
        "predictions":   preds,
        "aqi_labels":    labels,
        "probabilities": proba,
        "class_order":   AQI_LABELS,
    }


def output_fn(prediction: dict, accept: str = "application/json"):
    if accept in ("application/json", "*/*"):
        return _json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept: {accept}")


if __name__ == "__main__":
    main()