"""
glue/glue_manager.py
=====================
Handles: store API key → upload script → create/update job → run → poll.
"""

import json
import logging
import time
import boto3

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class GlueJobManager:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.glue   = boto3.client("glue",          region_name=config.region)
        self.s3     = boto3.client("s3",             region_name=config.region)
        self.sm_sec = boto3.client("secretsmanager", region_name=config.region)

    # ── 1. Store API key ──────────────────────────────────────

    def store_api_key(self, api_key: str) -> None:
        secret_value = json.dumps({"api_key": api_key})
        try:
            self.sm_sec.get_secret_value(SecretId=self.config.openaq_secret_name)
            self.sm_sec.update_secret(
                SecretId=self.config.openaq_secret_name,
                SecretString=secret_value,
            )
            logger.info("Secret updated: %s", self.config.openaq_secret_name)
        except self.sm_sec.exceptions.ResourceNotFoundException:
            self.sm_sec.create_secret(
                Name=self.config.openaq_secret_name,
                Description="OpenAQ v3 API key for India AQI pipeline",
                SecretString=secret_value,
            )
            logger.info("Secret created: %s", self.config.openaq_secret_name)

    # ── 2. Upload script ──────────────────────────────────────

    def upload_script(self, local_path: str = "glue/glue_etl.py") -> str:
        key = "glue/scripts/glue_etl.py"
        with open(local_path, "rb") as fh:
            self.s3.put_object(Bucket=self.config.s3_bucket, Key=key, Body=fh.read())
        uri = f"s3://{self.config.s3_bucket}/{key}"
        logger.info("Script uploaded: %s", uri)
        return uri

    # ── 3. Create or update Glue job ─────────────────────────

    def create_or_update_job(self) -> None:
        job_params = {
            "Name": self.config.glue_job_name,
            "Role": self.config.glue_role_arn,
            "Command": {
                "Name":           "glueetl",
                "ScriptLocation": self.config.glue_script_uri,
                "PythonVersion":  "3",
            },
            "DefaultArguments": {
                "--job-language":                     "python",
                "--enable-metrics":                   "true",
                "--enable-continuous-cloudwatch-log": "true",
                "--TempDir": f"s3://{self.config.s3_bucket}/glue/tmp/",
                "--S3_OUTPUT_PATH": self.config.processed_data_uri,
                "--LOOKBACK_DAYS":  str(self.config.lookback_days),
                "--SECRET_NAME":    self.config.openaq_secret_name,
            },
            "GlueVersion":     self.config.glue_version,
            "WorkerType":      self.config.glue_worker_type,
            "NumberOfWorkers": self.config.glue_num_workers,
            "Timeout":         120,
            "MaxRetries":      1,
            "ExecutionProperty": {"MaxConcurrentRuns": 1},
        }
        try:
            self.glue.get_job(JobName=self.config.glue_job_name)
            update = {k: v for k, v in job_params.items() if k != "Name"}
            self.glue.update_job(JobName=self.config.glue_job_name, JobUpdate=update)
            logger.info("Glue job updated: %s", self.config.glue_job_name)
        except self.glue.exceptions.EntityNotFoundException:
            self.glue.create_job(**job_params)
            logger.info("Glue job created: %s", self.config.glue_job_name)

    # ── 4. Start job run ──────────────────────────────────────

    def start_job_run(self) -> str:
        run_id = self.glue.start_job_run(
            JobName=self.config.glue_job_name,
            Arguments={
                "--S3_OUTPUT_PATH": self.config.processed_data_uri,
                "--SECRET_NAME":    self.config.openaq_secret_name,
                "--LOOKBACK_DAYS":  str(self.config.lookback_days),
            },
        )["JobRunId"]
        logger.info("Glue run started: %s", run_id)
        return run_id

    # ── 5. Poll until complete ────────────────────────────────

    def wait_for_completion(self, run_id: str, poll_interval: int = 30) -> str:
        terminal = {"SUCCEEDED", "FAILED", "ERROR", "STOPPED", "TIMEOUT"}
        while True:
            run   = self.glue.get_job_run(
                JobName=self.config.glue_job_name,
                RunId=run_id,
                PredecessorsIncluded=False,
            )["JobRun"]
            state = run["JobRunState"]
            logger.info("Glue state: %s  (%ds elapsed)", state, run.get("ExecutionTime", 0))
            if state == "SUCCEEDED":
                return state
            if state in terminal:
                raise RuntimeError(
                    f"Glue job {state}: {run.get('ErrorMessage', 'no details')}"
                )
            time.sleep(poll_interval)

    # ── Orchestrate ───────────────────────────────────────────

    def run_etl_pipeline(self) -> str:
        self.upload_script()
        self.create_or_update_job()
        run_id = self.start_job_run()
        self.wait_for_completion(run_id)
        logger.info("Glue ETL complete. Data at: %s", self.config.processed_data_uri)
        return self.config.processed_data_uri

    def ensure_glue_database(self) -> None:
        try:
            self.glue.get_database(Name=self.config.glue_database_name)
        except self.glue.exceptions.EntityNotFoundException:
            self.glue.create_database(DatabaseInput={"Name": self.config.glue_database_name})
            logger.info("Glue database created: %s", self.config.glue_database_name)