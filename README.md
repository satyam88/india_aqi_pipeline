# India Air Quality MLOps Pipeline

An end-to-end MLOps pipeline that fetches real-time air quality data from the OpenAQ v3 API, transforms it using AWS Glue, and trains a machine learning model using Amazon SageMaker AI to predict India CPCB AQI categories.

---

## Architecture

```
OpenAQ v3 API  (India air quality — 17 monitoring stations)
      │
      ▼
AWS Glue ETL
  fetch → clean → pivot → compute AQI label → S3 Parquet
      │
      ▼
Amazon S3
  s3://your-bucket/data/processed/
      │
      ▼
SageMaker Pipeline
  ├── PreprocessingStep   OHE + StandardScaler + 70/15/15 split
  ├── TrainingStep        RandomForestClassifier (300 trees, 6-class)
  ├── EvaluationStep      accuracy, F1, AUC → evaluation.json
  ├── ConditionStep       accuracy ≥ 0.78 → register or fail
  └── RegisterModelStep   model package in Model Registry
      │
      ▼
SageMaker Endpoint
  real-time inference → AQI category prediction
```

---

## Dataset

- **Source:** [OpenAQ v3 API](https://api.openaq.org/v3) — India (country_id=9)
- **Stations:** 17 government CPCB monitoring stations across India
- **Pollutants:** PM2.5, PM10, NO2, SO2, CO, O3
- **Target:** India CPCB AQI category (6 classes)

| Class | Label | PM2.5 (µg/m³) |
|---|---|---|
| 0 | Good | 0 – 30 |
| 1 | Satisfactory | 31 – 60 |
| 2 | Moderate | 61 – 90 |
| 3 | Poor | 91 – 120 |
| 4 | Very Poor | 121 – 250 |
| 5 | Severe | > 250 |

---

## Project Structure

```
india_aqi_pipeline/
├── config/
│   ├── __init__.py
│   └── pipeline_config.py      # all settings in one place
├── glue/
│   ├── __init__.py
│   ├── glue_etl.py             # Glue 4.0 ETL — API → S3
│   └── glue_manager.py         # deploy, run, poll Glue job
├── infra/
│   └── setup.sh                # one-time AWS infrastructure setup
├── pipeline/
│   ├── __init__.py
│   └── pipeline.py             # SageMaker Pipeline DAG + deploy
├── scripts/
│   ├── preprocessing.py        # ProcessingStep container script
│   ├── training.py             # TrainingStep + inference handlers
│   └── evaluation.py           # EvaluationStep container script
├── invoke_endpoint.py          # test the live endpoint
├── requirements.txt
└── run_pipeline.py             # CLI entry point
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 | via conda |
| AWS account | with SageMaker + Glue permissions |
| AWS CLI configured | `aws configure` |
| OpenAQ API key | free at [explore.openaq.org](https://explore.openaq.org) |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/satyam88/india_aqi_pipeline.git
cd india_aqi_pipeline
```

### 2. Create conda environment

```bash
conda create -n india_aqi python=3.10 -y
conda activate india_aqi
pip install -r requirements.txt
```

### 3. Configure AWS CLI

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (ap-south-1), output (json)
```

### 4. Run infrastructure setup

```bash
bash infra/setup.sh

# Copy and export the env vars printed at the end
export AWS_DEFAULT_REGION="ap-south-1"
export S3_BUCKET="india-aqi-mlops-<your-account-id>"
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account>:role/SageMakerAQIRole"
export GLUE_ROLE_ARN="arn:aws:iam::<account>:role/GlueAQIRole"
export OPENAQ_API_KEY="your-openaq-api-key"
```

---

## Running the Pipeline

### Run everything end to end

```bash
python run_pipeline.py --mode full
```

### Run each stage individually

```bash
# Stage 1 — fetch data from OpenAQ API and land in S3
python run_pipeline.py --mode glue

# Stage 2 — run SageMaker pipeline (preprocess → train → evaluate → register)
python run_pipeline.py --mode run

# Stage 3 — deploy the approved model to a real-time endpoint
python run_pipeline.py --mode deploy
```

### Approve the model before deploying

```bash
# Get model package ARN
aws sagemaker list-model-packages \
  --model-package-group-name IndiaAQIModelGroup \
  --region ap-south-1 \
  --query "ModelPackageSummaryList[0].ModelPackageArn" \
  --output text

# Approve it
aws sagemaker update-model-package \
  --model-package-arn <ARN> \
  --model-approval-status Approved \
  --region ap-south-1
```

---

## Testing the Endpoint

```bash
python invoke_endpoint.py
```

Example output:

```
Delhi rush hour  : Severe (75% confidence)
Bangalore morning: Good   (82% confidence)
Hyderabad evening: Moderate (71% confidence)
```

### Monitored stations

| Station | City |
|---|---|
| Anand Vihar, New Delhi - DPCC | Delhi |
| Income Tax Office, Delhi - DPCC | Delhi |
| IHBAS, Delhi - CPCB | Delhi |
| Mandir Marg, Delhi - DPCC | Delhi |
| Punjabi Bagh, Delhi - DPCC | Delhi |
| R K Puram, Delhi - DPCC | Delhi |
| Delhi Technological University, Delhi - CPCB | Delhi |
| Vikas Sadan, Gurugram - HSPCB | Gurugram |
| MD University, Rohtak - HSPCB | Rohtak |
| BTM Layout, Bengaluru - KSPCB | Bengaluru |
| Peenya, Bengaluru - KSPCB | Bengaluru |
| Lalbagh, DN Park | Bengaluru |
| Zoo Park, Hyderabad - TSPCB | Hyderabad |
| Alandur Bus Depot | Chennai |
| Collectorate - Gaya - BSPCB | Gaya |
| Collectorate Jodhpur - RSPCB | Jodhpur |
| Haldia, Haldia - WBPCB | Haldia |

---

## SageMaker Pipeline Steps

| Step | Type | What it does |
|---|---|---|
| PreprocessingStep | SKLearnProcessor | Reads Parquet, derives time features, OHE city, StandardScaler, 70/15/15 split |
| TrainingStep | SKLearn estimator | Trains RandomForestClassifier, logs metrics, saves model.joblib |
| EvaluationStep | SKLearnProcessor | Scores test set, writes evaluation.json |
| ConditionStep | Condition | accuracy ≥ 0.78 → register, else fail |
| RegisterModelStep | ModelStep | Registers model package in Model Registry |

---

## Cleanup

Delete all AWS resources when done to avoid charges:

```bash
# Delete endpoint (~$0.115/hour if left running)
aws sagemaker delete-endpoint \
  --endpoint-name india-aqi-endpoint \
  --region ap-south-1

# Delete pipeline
aws sagemaker delete-pipeline \
  --pipeline-name india-aqi-mlops-pipeline \
  --region ap-south-1

# Delete S3 bucket
aws s3 rb s3://${S3_BUCKET} --force
```

---

## Estimated Cost

| Resource | Cost |
|---|---|
| Glue ETL (3-4 runs) | ~$0.25 |
| SageMaker Processing + Training | ~$0.50 |
| SageMaker Endpoint (1 hour) | ~$0.12 |
| S3 + CloudWatch | ~$0.05 |
| **Total** | **~$1-2** |

---

## References

- [OpenAQ API v3 Documentation](https://docs.openaq.org)
- [Amazon SageMaker AI Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [AWS Glue Developer Guide](https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html)
- [India CPCB AQI Standards](https://cpcb.nic.in/air-quality-index/)