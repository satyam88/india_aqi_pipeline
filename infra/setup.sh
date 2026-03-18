#!/bin/bash
# infra/setup.sh
# ===============
# One-time AWS infrastructure setup.
# Run this before anything else.

set -euo pipefail

export AWS_REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET="india-aqi-mlops-${AWS_ACCOUNT}"
export SM_ROLE="SageMakerAQIRole"
export GLUE_ROLE="GlueAQIRole"

echo "Account : $AWS_ACCOUNT"
echo "Region  : $AWS_REGION"
echo "Bucket  : $S3_BUCKET"

# ── S3 bucket ─────────────────────────────────────────────────
aws s3 mb s3://${S3_BUCKET} --region ${AWS_REGION} || echo "Bucket already exists."

aws s3api put-bucket-versioning \
  --bucket ${S3_BUCKET} \
  --versioning-configuration Status=Enabled

aws s3api put-public-access-block \
  --bucket ${S3_BUCKET} \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

for PREFIX in data/raw data/processed glue/scripts glue/tmp; do
  aws s3api put-object --bucket ${S3_BUCKET} --key "${PREFIX}/"
done

echo "S3 ready: s3://${S3_BUCKET}"

# ── SageMaker IAM role ────────────────────────────────────────
aws iam create-role \
  --role-name ${SM_ROLE} \
  --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
  || echo "SageMaker role already exists."

aws iam attach-role-policy --role-name ${SM_ROLE} \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name ${SM_ROLE} \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

export SAGEMAKER_ROLE_ARN=$(aws iam get-role \
  --role-name ${SM_ROLE} --query Role.Arn --output text)
echo "SageMaker role: ${SAGEMAKER_ROLE_ARN}"

# ── Glue IAM role ─────────────────────────────────────────────
aws iam create-role \
  --role-name ${GLUE_ROLE} \
  --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"glue.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
  || echo "Glue role already exists."

aws iam attach-role-policy --role-name ${GLUE_ROLE} \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
aws iam attach-role-policy --role-name ${GLUE_ROLE} \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Allow Glue to read the OpenAQ API key from Secrets Manager
aws iam put-role-policy \
  --role-name ${GLUE_ROLE} \
  --policy-name SecretsManagerReadAccess \
  --policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["secretsmanager:GetSecretValue","secretsmanager:DescribeSecret"],"Resource":"arn:aws:secretsmanager:*:*:secret:openaq/*"}]}'

export GLUE_ROLE_ARN=$(aws iam get-role \
  --role-name ${GLUE_ROLE} --query Role.Arn --output text)
echo "Glue role: ${GLUE_ROLE_ARN}"

# ── Glue Data Catalog database ────────────────────────────────
aws glue create-database \
  --database-input '{"Name":"india_aqi_db","Description":"India AQI MLOps pipeline"}' \
  || echo "Glue database already exists."

# ── SageMaker Model Package Group ─────────────────────────────
aws sagemaker create-model-package-group \
  --model-package-group-name IndiaAQIModelGroup \
  --model-package-group-description "India AQI 6-class AQI classifier" \
  || echo "Model package group already exists."

# ── Print env vars ────────────────────────────────────────────
cat <<EOF

# Copy and run these in your terminal:
export AWS_DEFAULT_REGION="${AWS_REGION}"
export S3_BUCKET="${S3_BUCKET}"
export SAGEMAKER_ROLE_ARN="${SAGEMAKER_ROLE_ARN}"
export GLUE_ROLE_ARN="${GLUE_ROLE_ARN}"

EOF

echo "Setup complete."