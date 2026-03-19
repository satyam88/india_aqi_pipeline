#!/bin/bash
# infra/teardown.sh
# ==================
# Deletes all AWS resources created by setup.sh
# Run this when the lab is done to stop all charges.

set -euo pipefail

export AWS_REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET="india-aqi-mlops-${AWS_ACCOUNT}"

echo "Account : $AWS_ACCOUNT"
echo "Region  : $AWS_REGION"
echo "Bucket  : $S3_BUCKET"
echo ""
echo "WARNING: This will delete everything. Press Ctrl+C to cancel."
echo "Continuing in 5 seconds..."
sleep 5

# ── SageMaker Endpoint ────────────────────────────────────────
echo "Deleting endpoint..."
aws sagemaker delete-endpoint \
  --endpoint-name india-aqi-endpoint \
  --region ${AWS_REGION} 2>/dev/null && echo "  Endpoint deleted." || echo "  Endpoint not found — skipping."

aws sagemaker delete-endpoint-config \
  --endpoint-config-name india-aqi-endpoint \
  --region ${AWS_REGION} 2>/dev/null && echo "  Endpoint config deleted." || echo "  Endpoint config not found — skipping."

# ── SageMaker Pipeline ────────────────────────────────────────
echo "Deleting pipeline..."
aws sagemaker delete-pipeline \
  --pipeline-name india-aqi-mlops-pipeline \
  --region ${AWS_REGION} 2>/dev/null && echo "  Pipeline deleted." || echo "  Pipeline not found — skipping."

# ── SageMaker Model Packages ──────────────────────────────────
echo "Deleting model packages..."
ARNS=$(aws sagemaker list-model-packages \
  --model-package-group-name IndiaAQIModelGroup \
  --region ${AWS_REGION} \
  --query "ModelPackageSummaryList[*].ModelPackageArn" \
  --output text 2>/dev/null)

for ARN in $ARNS; do
  aws sagemaker delete-model-package \
    --model-package-name "$ARN" \
    --region ${AWS_REGION} 2>/dev/null && echo "  Deleted: $ARN" || true
done

aws sagemaker delete-model-package-group \
  --model-package-group-name IndiaAQIModelGroup \
  --region ${AWS_REGION} 2>/dev/null && echo "  Model package group deleted." || echo "  Model package group not found — skipping."

# ── Glue Job ──────────────────────────────────────────────────
echo "Deleting Glue job..."
aws glue delete-job \
  --job-name india-aqi-glue-etl \
  --region ${AWS_REGION} 2>/dev/null && echo "  Glue job deleted." || echo "  Glue job not found — skipping."

# ── Glue Database ─────────────────────────────────────────────
echo "Deleting Glue database..."
aws glue delete-database \
  --name india_aqi_db \
  --region ${AWS_REGION} 2>/dev/null && echo "  Glue database deleted." || echo "  Glue database not found — skipping."

# ── S3 Buckets ────────────────────────────────────────────────
echo "Deleting S3 buckets..."
aws s3 rb s3://${S3_BUCKET} --force 2>/dev/null && echo "  Deleted: s3://${S3_BUCKET}" || echo "  Bucket not found — skipping."

aws s3 rb s3://sagemaker-${AWS_REGION}-${AWS_ACCOUNT} --force 2>/dev/null && echo "  Deleted: s3://sagemaker-${AWS_REGION}-${AWS_ACCOUNT}" || echo "  SageMaker bucket not found — skipping."

# ── IAM Roles ─────────────────────────────────────────────────
echo "Deleting IAM roles..."

for POLICY in AmazonSageMakerFullAccess AmazonS3FullAccess; do
  aws iam detach-role-policy \
    --role-name SageMakerAQIRole \
    --policy-arn arn:aws:iam::aws:policy/${POLICY} 2>/dev/null || true
done

aws iam delete-role \
  --role-name SageMakerAQIRole 2>/dev/null && echo "  SageMaker role deleted." || echo "  SageMaker role not found — skipping."

for POLICY in AWSGlueServiceRole AmazonS3FullAccess; do
  aws iam detach-role-policy \
    --role-name GlueAQIRole \
    --policy-arn arn:aws:iam::aws:policy/${POLICY} 2>/dev/null || true
  aws iam detach-role-policy \
    --role-name GlueAQIRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/${POLICY} 2>/dev/null || true
done

aws iam delete-role-policy \
  --role-name GlueAQIRole \
  --policy-name SecretsManagerReadAccess 2>/dev/null || true

aws iam delete-role \
  --role-name GlueAQIRole 2>/dev/null && echo "  Glue role deleted." || echo "  Glue role not found — skipping."

# ── CloudWatch Log Groups ─────────────────────────────────────
echo "Deleting CloudWatch log groups..."
for LG in \
  "/aws/sagemaker/Endpoints/india-aqi-endpoint" \
  "/aws/sagemaker/ProcessingJobs" \
  "/aws/sagemaker/TrainingJobs" \
  "/aws-glue/jobs/output"; do
  aws logs delete-log-group \
    --log-group-name "$LG" \
    --region ${AWS_REGION} 2>/dev/null && echo "  Deleted: $LG" || true
done

echo ""
echo "Teardown complete. All resources deleted."
