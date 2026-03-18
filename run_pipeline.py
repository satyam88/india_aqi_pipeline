"""
run_pipeline.py
================
CLI entry point.

Modes:
  store-key  — save OpenAQ API key to AWS Secrets Manager
  glue       — run the Glue ETL job (API → S3)
  run        — run the SageMaker pipeline (preprocess → train → evaluate → register)
  deploy     — deploy the latest Approved model to an endpoint
  full       — glue → run → deploy in one command

Examples:
  python run_pipeline.py --mode store-key --api-key YOUR_KEY
  python run_pipeline.py --mode glue
  python run_pipeline.py --mode run
  python run_pipeline.py --mode deploy
  python run_pipeline.py --mode full
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.pipeline_config import PipelineConfig
from glue.glue_manager import GlueJobManager
from pipeline.pipeline import run, deploy_approved_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="India AQI MLOps Pipeline")
    p.add_argument("--mode", choices=["store-key", "glue", "run", "deploy", "full"], default="full")
    p.add_argument("--api-key",            type=str,   default=None)
    p.add_argument("--deploy",             action="store_true")
    p.add_argument("--config",             type=str,   default=None)
    p.add_argument("--s3-bucket",          type=str,   default=None)
    p.add_argument("--region",             type=str,   default=None)
    p.add_argument("--role-arn",           type=str,   default=None)
    p.add_argument("--glue-role-arn",      type=str,   default=None)
    p.add_argument("--accuracy-threshold", type=float, default=None)
    p.add_argument("--lookback-days",      type=int,   default=None)
    return p.parse_args()


def build_config(args) -> PipelineConfig:
    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    overrides = {
        "s3_bucket":          args.s3_bucket,
        "region":             args.region,
        "role_arn":           args.role_arn,
        "glue_role_arn":      args.glue_role_arn,
        "accuracy_threshold": args.accuracy_threshold,
        "lookback_days":      args.lookback_days,
    }
    for attr, val in overrides.items():
        if val is not None:
            setattr(config, attr, val)
    config.validate()
    return config


def main():
    args   = parse_args()
    config = build_config(args)
    glue   = GlueJobManager(config)

    logger.info("Mode     : %s", args.mode)
    logger.info("S3 bucket: %s", config.s3_bucket)
    logger.info("Region   : %s", config.region)

    if args.mode == "store-key":
        if not args.api_key:
            raise ValueError("--api-key is required with --mode store-key")
        glue.store_api_key(args.api_key)
        return

    if args.mode in ("glue", "full"):
        glue.ensure_glue_database()
        glue.run_etl_pipeline()

    if args.mode in ("run", "full"):
        deploy = args.deploy or args.mode == "full"
        run(config, execute=True, deploy=deploy)

    if args.mode == "deploy":
        deploy_approved_model(config)


if __name__ == "__main__":
    main()