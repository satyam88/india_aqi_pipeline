# India Air Quality MLOps Pipeline

An end-to-end MLOps pipeline that fetches real-time air quality data from the OpenAQ v3 API, transforms it using AWS Glue, and trains a machine learning model using Amazon SageMaker to predict India CPCB AQI categories.

---

## Architecture

```
OpenAQ v3 API  (India air quality — 17 CPCB monitoring stations)
      │
      ▼
AWS Glue ETL
  fetch → clean → pivot → compute AQI label → S3 Parquet
      │
      ▼
Amazon S3
  s3://<your-bucket>/data/processed/
      │
      ▼
SageMaker Pipeline
  ├── PreprocessingStep   OHE + StandardScaler + 70/15/15 split → saves scaler.joblib + columns.json
  ├── TrainingStep        RandomForestClassifier (300 trees, 6-class) → packages model.tar.gz
  ├── EvaluationStep      accuracy, F1, AUC → evaluation.json
  ├── ConditionStep       accuracy ≥ 0.78 → register, else fail
  └── RegisterModelStep   model package in Model Registry
      │
      ▼
SageMaker Endpoint
  real-time inference → AQI category + probabilities
```

> **Important:** Pass real pollutant readings (`--pm25`, `--pm10`, etc.) when invoking the endpoint for accurate predictions. Without them the model falls back to time and location features only.

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
│   ├── glue_etl.py             # Glue 4.0 ETL — API → S3 Parquet
│   └── glue_manager.py         # deploy, run, poll Glue job
├── infra/
│   └── setup.sh                # one-time AWS infrastructure setup
├── pipeline/
│   ├── __init__.py
│   └── pipeline.py             # SageMaker Pipeline DAG + deploy logic
├── scripts/
│   ├── preprocess.py           # ProcessingStep — OHE, scale, split, saves scaler + columns
│   ├── training.py             # TrainingStep + inference handlers (model_fn, input_fn, predict_fn)
│   └── evaluation.py           # EvaluationStep — scores test set, writes evaluation.json
├── invoke_endpoint.py          # CLI to test the live endpoint
├── requirements.txt
└── run_pipeline.py             # CLI entry point
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 | via conda or venv |
| AWS account | SageMaker + Glue + S3 permissions required |
| AWS CLI configured | `aws configure` |
| OpenAQ API key | free at [explore.openaq.org](https://explore.openaq.org) |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/satyam88/india_aqi_pipeline.git
cd india_aqi_pipeline
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins `sagemaker>=2.200.0,<3.0.0`. Do not upgrade to sagemaker v3 — the `workflow` module API changed significantly.

### 3. Configure AWS CLI

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (ap-south-1), output format (json)
```

### 4. Run infrastructure setup

```bash
bash infra/setup.sh

# Export the env vars printed at the end of setup
export AWS_DEFAULT_REGION="ap-south-1"
export S3_BUCKET="india-aqi-mlops-<your-account-id>"
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account>:role/SageMakerAQIRole"
export GLUE_ROLE_ARN="arn:aws:iam::<account>:role/GlueAQIRole"
export OPENAQ_API_KEY="your-openaq-api-key"
```

### 5. Store the OpenAQ API key in Secrets Manager

```bash
python run_pipeline.py --mode store-key --api-key YOUR_OPENAQ_KEY
```

---

## Running the Pipeline

### Full end-to-end

```bash
python run_pipeline.py --mode full
```

### Stage by stage

```bash
# Stage 1 — fetch data from OpenAQ API and land in S3 as Parquet
python run_pipeline.py --mode glue

# Stage 2 — preprocess → train → evaluate → register model
python run_pipeline.py --mode run

# Stage 3 — approve model, then deploy to real-time endpoint
python run_pipeline.py --mode deploy
```

---

## Approving and Deploying the Model

The ConditionStep registers the model only if `accuracy ≥ 0.78`. You must approve it in Model Registry before deploying.

```bash
# 1. Get the latest model package ARN
aws sagemaker list-model-packages \
  --model-package-group-name IndiaAQIModelGroup \
  --region ap-south-1 \
  --query "ModelPackageSummaryList[0].ModelPackageArn" \
  --output text

# 2. Approve it (replace VERSION_NUMBER)
aws sagemaker update-model-package \
  --model-package-arn arn:aws:sagemaker:ap-south-1:<account>:model-package/IndiaAQIModelGroup/VERSION_NUMBER \
  --model-approval-status Approved \
  --region ap-south-1

# 3. If redeploying, delete the existing endpoint config first
aws sagemaker delete-endpoint-config \
  --endpoint-config-name india-aqi-endpoint \
  --region ap-south-1

# 4. Deploy
python run_pipeline.py --mode deploy
```

---

## Testing the Endpoint

### With time and location only

```bash
python invoke_endpoint.py --location Delhi --datetime "2026-01-15T19:00:00"
```

### With real pollutant readings (recommended for accurate predictions)

```bash
python invoke_endpoint.py \
  --location Delhi \
  --datetime "2026-01-15T19:00:00" \
  --pm25 250.0 --pm10 350.0 --no2 120.0 --so2 50.0
```

### Example output

```
── Prediction Result ─────────────────────────────────────
  AQI Category : Very Poor
  Class Index  : 4
  Class Order  : ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

  Probabilities:
    Good            0.0000
    Satisfactory    0.0070  
    Moderate        0.0167  
    Poor            0.0504  █
    Very Poor       0.8247  ████████████████████████
    Severe          0.1012  ███
──────────────────────────────────────────────────────────
```

> Without pollutant values the model defaults all readings to `-1` (missing sentinel) and relies on time + location features only — predictions will skew toward **Good**. Always pass real sensor readings for meaningful results.

---

## Monitored Stations

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
| PreprocessingStep | SKLearnProcessor | Reads Parquet, derives time features, one-hot encodes city, applies StandardScaler, splits 70/15/15, saves `scaler.joblib` + `columns.json` |
| TrainingStep | SKLearn estimator | Trains RandomForestClassifier, copies scaler + columns into `model.tar.gz` alongside `model.joblib` |
| EvaluationStep | SKLearnProcessor | Scores test set, writes `evaluation.json` with accuracy, F1, AUC |
| ConditionStep | Condition | If accuracy ≥ 0.78 → RegisterModelStep, else pipeline fails |
| RegisterModelStep | ModelStep | Registers model package in Model Registry with `PendingManualApproval` status |

---

## Inference Contract

The endpoint accepts `application/json` with two payload formats:

**High-level (recommended):**
```json
{
  "location": "Delhi",
  "datetime": "2026-01-15T19:00:00",
  "pm25": 250.0,
  "pm10": 350.0,
  "no2": 120.0,
  "so2": 50.0,
  "co": 1.2,
  "o3": 30.0
}
```

**Raw numeric (legacy):**
```json
{ "instances": [[pm25, pm10, no2, so2, co, o3, hour, month, is_rush_hour, is_weekend, city_Delhi, ...]] }
```

**Response:**
```json
{
  "predictions":   [4],
  "aqi_labels":    ["Very Poor"],
  "probabilities": [[0.0, 0.007, 0.016, 0.05, 0.825, 0.101]],
  "class_order":   ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
}
```

---

## Cleanup

Delete all AWS resources when done to avoid charges:

```bash
# Delete endpoint (~$0.115/hour if left running)
aws sagemaker delete-endpoint \
  --endpoint-name india-aqi-endpoint \
  --region ap-south-1

# Delete endpoint config
aws sagemaker delete-endpoint-config \
  --endpoint-config-name india-aqi-endpoint \
  --region ap-south-1

# Delete pipeline
aws sagemaker delete-pipeline \
  --pipeline-name india-aqi-mlops-pipeline \
  --region ap-south-1

# Delete S3 bucket (irreversible)
aws s3 rb s3://${S3_BUCKET} --force
```

---

## Estimated Cost

| Resource | Cost |
|---|---|
| Glue ETL (3–4 runs) | ~$0.25 |
| SageMaker Processing + Training | ~$0.50 |
| SageMaker Endpoint (1 hour) | ~$0.12 |
| S3 + CloudWatch Logs | ~$0.05 |
| **Total** | **~$1–2** |

---

## References

- [OpenAQ API v3 Documentation](https://docs.openaq.org)
- [Amazon SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [AWS Glue Developer Guide](https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html)
- [India CPCB AQI Standards](https://cpcb.nic.in/air-quality-index/)