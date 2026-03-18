"""
config/pipeline_config.py
==========================
Single source of truth for every setting in the pipeline.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:

    # ── Identity ──────────────────────────────────────────────
    pipeline_name:        str = "india-aqi-mlops-pipeline"
    model_package_group:  str = "IndiaAQIModelGroup"
    endpoint_name:        str = "india-aqi-endpoint"
    glue_job_name:        str = "india-aqi-glue-etl"
    glue_database_name:   str = "india_aqi_db"
    glue_table_name:      str = "aqi_readings_processed"

    # ── AWS ───────────────────────────────────────────────────
    region: str = field(
        default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    )
    role_arn: Optional[str] = field(
        default_factory=lambda: os.environ.get("SAGEMAKER_ROLE_ARN")
    )
    glue_role_arn: Optional[str] = field(
        default_factory=lambda: os.environ.get("GLUE_ROLE_ARN")
    )

    # ── S3 ────────────────────────────────────────────────────
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get("S3_BUCKET", "your-india-aqi-bucket")
    )

    @property
    def glue_script_uri(self) -> str:
        return f"s3://{self.s3_bucket}/glue/scripts/glue_etl.py"

    @property
    def processed_data_uri(self) -> str:
        return f"s3://{self.s3_bucket}/data/processed/"

    @property
    def input_data_uri(self) -> str:
        return self.processed_data_uri

    # ── OpenAQ ────────────────────────────────────────────────
    openaq_base_url:    str = "https://api.openaq.org/v3"
    openaq_secret_name: str = "openaq/api-key"
    lookback_days:      int = 30

    # ── Splits ────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15

    # ── Instances ─────────────────────────────────────────────
    processing_instance_type: str = "ml.m5.xlarge"
    training_instance_type:   str = "ml.m5.xlarge"
    inference_instance_type:  str = "ml.m5.large"
    endpoint_instance_count:  int = 1

    # ── Glue ──────────────────────────────────────────────────
    glue_worker_type: str = "G.1X"
    glue_num_workers: int = 2
    glue_version:     str = "4.0"

    # ── Hyperparameters ───────────────────────────────────────
    n_estimators:      int = 300
    max_depth:         int = 15
    min_samples_split: int = 5
    class_weight:      str = "balanced"

    # ── Quality gate ──────────────────────────────────────────
    accuracy_threshold: float = 0.78

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as fh:
            data = json.load(fh)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate(self) -> None:
        if round(self.train_ratio + self.val_ratio + self.test_ratio, 6) != 1.0:
            raise ValueError("train/val/test ratios must sum to 1.0")
        if self.s3_bucket == "your-india-aqi-bucket":
            raise ValueError("Set S3_BUCKET env var before running.")