cat > scripts/preprocessing.py << 'EOF'
"""
scripts/preprocessing.py
=========================
Runs inside SageMaker SKLearnProcessor container.
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "-q"])

import argparse
import glob
import json
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INPUT_DIR  = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"
TARGET_COL = "aqi_category"

DROP_COLS = ["location_id", "location_name", "timestamp_utc", "ingestion_date"]
NUM_COLS  = ["pm25", "pm10", "no2", "so2", "co", "o3"]
CAT_COLS  = ["city"]


def load_parquet(input_dir: str) -> pd.DataFrame:
    # Search all levels
    files = glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True)

    # Also check root level
    files += glob.glob(os.path.join(input_dir, "*.parquet"))

    # Deduplicate
    files = list(set(files))

    logger.info("Input directory contents:")
    for root, dirs, fs in os.walk(input_dir):
        for f in fs:
            logger.info("  %s", os.path.join(root, f))

    if not files:
        raise FileNotFoundError(f"No Parquet files found under {input_dir}")

    logger.info("Loading %d Parquet files", len(files))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info("Loaded %d rows", len(df))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["hour_of_day"]   = df["timestamp_utc"].dt.hour
    df["month"]         = df["timestamp_utc"].dt.month
    df["is_rush_hour"]  = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_weekend"]    = (df["timestamp_utc"].dt.dayofweek >= 5).astype(int)
    return df


def encode_and_scale(X_train, X_val, X_test):
    cat_present = [c for c in CAT_COLS if c in X_train.columns]
    X_train = pd.get_dummies(X_train, columns=cat_present, drop_first=False)
    X_val   = pd.get_dummies(X_val,   columns=cat_present, drop_first=False)
    X_test  = pd.get_dummies(X_test,  columns=cat_present, drop_first=False)

    X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    num_present = [c for c in NUM_COLS if c in X_train.columns]
    scaler = StandardScaler()
    X_train[num_present] = scaler.fit_transform(X_train[num_present])
    X_val[num_present]   = scaler.transform(X_val[num_present])
    X_test[num_present]  = scaler.transform(X_test[num_present])

    logger.info("Feature shape — train: %s  val: %s  test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test


def save_split(X, y, name, filename):
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    out = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    path = os.path.join(out_dir, filename)
    out.to_csv(path, index=False)
    logger.info("Saved %s: %d rows → %s", name, len(out), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--test-ratio",  type=float, default=0.15)
    args = parser.parse_args()

    df = load_parquet(INPUT_DIR)
    df = shuffle(df, random_state=42).reset_index(drop=True)
    df = add_time_features(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    logger.info("Rows ready for training: %d", len(df))
    logger.info("Columns: %s", list(df.columns))
    logger.info("AQI distribution:\n%s", df[TARGET_COL].value_counts().sort_index().to_string())

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    test_size = args.test_ratio
    val_size  = args.val_ratio / (args.train_ratio + args.val_ratio)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=42, stratify=y_tmp
    )

    X_train, X_val, X_test = encode_and_scale(X_train, X_val, X_test)

    save_split(X_train, y_train, "train",      "train.csv")
    save_split(X_val,   y_val,   "validation", "validation.csv")
    save_split(X_test,  y_test,  "test",       "test.csv")

    # Save column order for inference
    col_path = os.path.join(OUTPUT_DIR, "train", "columns.json")
    with open(col_path, "w") as f:
        json.dump(list(X_train.columns), f)
    logger.info("columns.json saved.")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
EOF