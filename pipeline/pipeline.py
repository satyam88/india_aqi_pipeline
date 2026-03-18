"""
pipeline/pipeline.py
=====================
SageMaker Pipeline DAG:

  PreprocessingStep
       ↓
  TrainingStep
       ↓
  EvaluationStep
       ↓
  ConditionStep (accuracy >= threshold)
      YES → RegisterModelStep
      NO  → FailStep

  deploy_approved_model() — called separately after manual approval
"""

import logging
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model import Model
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.model_metrics import MetricsSource, ModelMetrics

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_pipeline(config: PipelineConfig) -> Pipeline:
    session = PipelineSession()
    role    = config.role_arn or sagemaker.get_execution_role()

    # ── Parameters ────────────────────────────────────────────
    p_instance    = ParameterString("ProcessingInstanceType", config.processing_instance_type)
    t_instance    = ParameterString("TrainingInstanceType",   config.training_instance_type)
    input_uri     = ParameterString("InputDataUri",           config.input_data_uri)
    approval      = ParameterString("ModelApprovalStatus",    "PendingManualApproval")
    acc_threshold = ParameterFloat ("AccuracyThreshold",      config.accuracy_threshold)

    # ── Step 1 · Preprocessing ────────────────────────────────
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=p_instance,
        instance_count=1,
        role=role,
        sagemaker_session=session,
        base_job_name=f"{config.pipeline_name}-preprocess",
    )

    preprocess_step = ProcessingStep(
        name="PreprocessingStep",
        processor=processor,
        inputs=[ProcessingInput(source=input_uri, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train",      source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
            ProcessingOutput(output_name="test",       source="/opt/ml/processing/output/test"),
        ],
        code="scripts/preprocessing.py",
        job_arguments=[
            "--train-ratio", str(config.train_ratio),
            "--val-ratio",   str(config.val_ratio),
            "--test-ratio",  str(config.test_ratio),
        ],
    )

    # ── Step 2 · Training ─────────────────────────────────────
    estimator = SKLearn(
        entry_point="training.py",
        framework_version="1.2-1",
        source_dir="scripts",
        instance_type=t_instance,
        instance_count=1,
        role=role,
        sagemaker_session=session,
        base_job_name=f"{config.pipeline_name}-training",
        hyperparameters={
            "n-estimators":      config.n_estimators,
            "max-depth":         config.max_depth,
            "min-samples-split": config.min_samples_split,
            "class-weight":      config.class_weight,
        },
        metric_definitions=[
            {"Name": "validation:accuracy", "Regex": "validation accuracy: ([0-9\\.]+)"},
            {"Name": "validation:f1",       "Regex": "validation f1: ([0-9\\.]+)"},
        ],
        environment={"OPENAQ_API_KEY": "{{resolve:secretsmanager:openaq/api-key:SecretString:api_key}}"},
    )

    training_step = TrainingStep(
        name="TrainingStep",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig
                    .Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig
                    .Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ── Step 3 · Evaluation ───────────────────────────────────
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=p_instance,
        instance_count=1,
        role=role,
        sagemaker_session=session,
        base_job_name=f"{config.pipeline_name}-evaluation",
    )

    eval_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_step = ProcessingStep(
        name="EvaluationStep",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig
                    .Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code="scripts/evaluation.py",
        property_files=[eval_report],
    )

    # ── Step 4 · Register ─────────────────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                eval_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    model = Model(
        image_uri=estimator.training_image_uri(),
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=session,
        role=role,
    )

    register_step = ModelStep(
        name="RegisterModelStep",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=[config.inference_instance_type],
            transform_instances=[config.inference_instance_type],
            model_package_group_name=config.model_package_group,
            approval_status=approval,
            model_metrics=model_metrics,
            description="India AQI 6-class classifier — RandomForest on OpenAQ v3 data",
        ),
    )

    # ── Step 4b · Fail ────────────────────────────────────────
    fail_step = FailStep(
        name="AccuracyGateFailed",
        error_message=Join(
            on=" ",
            values=["Accuracy below threshold:", acc_threshold],
        ),
    )

    # ── Step 5 · Condition ────────────────────────────────────
    condition_step = ConditionStep(
        name="AccuracyConditionStep",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_report,
                    json_path="metrics.accuracy.value",
                ),
                right=acc_threshold,
            )
        ],
        if_steps=[register_step],
        else_steps=[fail_step],
    )

    return Pipeline(
        name=config.pipeline_name,
        parameters=[p_instance, t_instance, input_uri, approval, acc_threshold],
        steps=[preprocess_step, training_step, eval_step, condition_step],
        sagemaker_session=session,
    )


def deploy_approved_model(config: PipelineConfig) -> str:
    from sagemaker.sklearn.model import SKLearnModel

    sm      = boto3.client("sagemaker", region_name=config.region)
    session = sagemaker.Session(boto_session=boto3.Session(region_name=config.region))
    role    = config.role_arn or sagemaker.get_execution_role()

    packages = sm.list_model_packages(
        ModelPackageGroupName=config.model_package_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    ).get("ModelPackageSummaryList", [])

    if not packages:
        raise RuntimeError(f"No Approved model packages in: {config.model_package_group}")

    # Get model data S3 URI from the package
    package_arn = packages[0]["ModelPackageArn"]
    package_details = sm.describe_model_package(ModelPackageName=package_arn)
    model_data = package_details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    logger.info("Model data: %s", model_data)

    model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point="training.py",
        source_dir="scripts",
        framework_version="1.2-1",
        sagemaker_session=session,
        env={"OPENAQ_API_KEY": "{{resolve:secretsmanager:openaq/api-key:SecretString:api_key}}"},
    )

    model.deploy(
        initial_instance_count=config.endpoint_instance_count,
        instance_type=config.inference_instance_type,
        endpoint_name=config.endpoint_name,
        wait=True,
    )

    logger.info("Endpoint live: %s", config.endpoint_name)
    return config.endpoint_name





def run(config: PipelineConfig, execute: bool = True, deploy: bool = False) -> None:
    role     = config.role_arn or sagemaker.get_execution_role()
    pipeline = create_pipeline(config)
    pipeline.upsert(role_arn=role)
    logger.info("Pipeline upserted: %s", config.pipeline_name)

    if execute:
        execution = pipeline.start(
            parameters={
                "InputDataUri":     config.input_data_uri,
                "AccuracyThreshold": config.accuracy_threshold,
            }
        )
        logger.info("Execution started: %s", execution.arn)
        execution.wait()
        steps = {s["StepName"]: s["StepStatus"] for s in execution.list_steps()}
        logger.info("Results: %s", steps)

    if deploy:
        deploy_approved_model(config)