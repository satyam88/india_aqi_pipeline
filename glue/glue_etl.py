"""
glue/glue_etl.py
=================
Glue 4.0 | Spark 3.3 | Python 3.10

Six responsibilities only:
  2. fetch_from_openaq — paginated GET with 429 retry
  3. clean_values      — drop nulls, negatives, zero readings
  4. pivot_wide        — long → wide (one row per location x hour)
  5. compute_aqi_label — CPCB PM2.5 breakpoints → category 0-5
  6. write_parquet     — write to S3 partitioned by ingestion_date

Glue job arguments:
  --S3_OUTPUT_PATH   s3://bucket/data/processed/
  --LOOKBACK_DAYS    30
  --SECRET_NAME      openaq/api-key
"""

import sys
import json
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

import boto3
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)

args = getResolvedOptions(sys.argv, ["JOB_NAME", "S3_OUTPUT_PATH"])
for key, default in [("LOOKBACK_DAYS", "30"), ("SECRET_NAME", "openaq/api-key")]:
    args[key] = getResolvedOptions(sys.argv, [key])[key] if f"--{key}" in sys.argv else default

job.init(args["JOB_NAME"], args)

BASE_URL         = "https://api.openaq.org/v3"
INDIA_COUNTRY_ID = 9
POLLUTANTS       = ["pm25", "pm10", "no2", "so2", "co", "o3"]

RAW_SCHEMA = StructType([
    StructField("location_id",   IntegerType(), True),
    StructField("location_name", StringType(),  True),
    StructField("city",          StringType(),  True),
    StructField("parameter",     StringType(),  True),
    StructField("value",         DoubleType(),  True),
    StructField("timestamp_utc", StringType(),  True),
])


# ── 1. get_api_key ────────────────────────────────────────────

def get_api_key(secret_name: str) -> str:
    import os
    if os.environ.get("OPENAQ_API_KEY"):
        logger.info("Using API key from environment variable.")
        return os.environ["OPENAQ_API_KEY"]
    logger.info("Fetching API key from Secrets Manager: %s", secret_name)
    secret = boto3.client("secretsmanager", region_name="ap-south-1").get_secret_value(
        SecretId=secret_name
    )["SecretString"]
    key = json.loads(secret).get("api_key", secret.strip('"'))
    logger.info("API key retrieved. Length: %d", len(key))
    return key


# ── 2. fetch_from_openaq ──────────────────────────────────────

def fetch_from_openaq(api_key: str, lookback_days: int) -> list:

    def get(path, params=None, retries=3):
        query = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
        url   = BASE_URL + path + query
        logger.info("GET %s", url)
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"X-API-Key": api_key, "Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                    logger.info("Response: found=%s results=%d",
                                data.get("meta", {}).get("found", "?"),
                                len(data.get("results", [])))
                    return data
            except urllib.error.HTTPError as e:
                logger.error("HTTP %d for %s: %s", e.code, url, e.reason)
                if e.code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f"All retries exhausted for {url}")

    def paginate(path, params, max_pages=1):
        """max_pages=1 means fetch only the first page per sensor — enough for a demo."""
        results, page = [], 1
        while page <= max_pages:
            params.update({"page": page, "limit": 1000})
            data  = get(path, params)
            batch = data.get("results", [])
            results.extend(batch)
            found = data.get("meta", {}).get("found", 0)
            if isinstance(found, str):
                found = 999
            if len(results) >= found or not batch:
                break
            page += 1
        return results

    date_from = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to   =  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("Fetching locations for India (country_id=%d)", INDIA_COUNTRY_ID)

    locations = paginate("/locations", {"countries_id": INDIA_COUNTRY_ID, "limit": 100})
    logger.info("Total India locations found: %d", len(locations))

    if not locations:
        logger.error("No locations found. Check API key and country ID.")
        return []

    records = []
    for loc in locations[:20]:   # 20 locations is enough for a demo
        logger.info("Processing location: %s (id=%s)", loc.get("name"), loc.get("id"))
        for sensor in loc.get("sensors", []):
            param = sensor["parameter"]["name"]
            if param not in POLLUTANTS:
                continue
            try:
                # max_pages=1 — fetch only first 1000 measurements per sensor
                measurements = paginate(
                    f"/sensors/{sensor['id']}/measurements/hourly",
                    {"date_from": date_from, "date_to": date_to},
                    max_pages=1,
                )
                logger.info("  sensor %d (%s): %d measurements", sensor["id"], param, len(measurements))
                for m in measurements:
                    records.append({
                        "location_id":   loc["id"],
                        "location_name": loc.get("name", ""),
                        "city":          loc.get("locality") or loc.get("name", ""),
                        "parameter":     param,
                        "value":         m.get("value"),
                        "timestamp_utc": m.get("period", {}).get("datetimeTo", {}).get("utc"),
                    })
            except Exception as exc:
                logger.warning("Skipping sensor %d (%s): %s", sensor["id"], param, exc)

    logger.info("Total raw records fetched: %d", len(records))
    return records

    def get(path, params=None, retries=3):
        query = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
        url   = BASE_URL + path + query
        logger.info("GET %s", url)
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"X-API-Key": api_key, "Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                    logger.info("Response: found=%s results=%d",
                                data.get("meta", {}).get("found", "?"),
                                len(data.get("results", [])))
                    return data
            except urllib.error.HTTPError as e:
                logger.error("HTTP %d for %s: %s", e.code, url, e.reason)
                if e.code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f"All retries exhausted for {url}")

    def paginate(path, params):
        results, page = [], 1
        while True:
            params.update({"page": page, "limit": 100})
            data  = get(path, params)
            batch = data.get("results", [])
            results.extend(batch)
            found = data.get("meta", {}).get("found", 0)
            if isinstance(found, str):
                found = 999
            if len(results) >= found or not batch:
                break
            page += 1
        return results

    date_from = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to   =  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("Fetching locations for India (country_id=%d)", INDIA_COUNTRY_ID)

    # Removed isMonitor filter — fetch all location types
    locations = paginate("/locations", {
        "countries_id": INDIA_COUNTRY_ID,
        "limit": 100,
    })

    logger.info("Total India locations found: %d", len(locations))

    if not locations:
        logger.error("No locations found. Check API key and country ID.")
        return []

    records = []
    for loc in locations[:50]:  # cap at 50 locations for first run
        logger.info("Processing location: %s (id=%s)", loc.get("name"), loc.get("id"))
        for sensor in loc.get("sensors", []):
            param = sensor["parameter"]["name"]
            if param not in POLLUTANTS:
                continue
            try:
                measurements = paginate(
                    f"/sensors/{sensor['id']}/measurements/hourly",
                    {"date_from": date_from, "date_to": date_to},
                )
                logger.info("  sensor %d (%s): %d measurements", sensor["id"], param, len(measurements))
                for m in measurements:
                    records.append({
                        "location_id":   loc["id"],
                        "location_name": loc.get("name", ""),
                        "city":          loc.get("locality") or loc.get("name", ""),
                        "parameter":     param,
                        "value":         m.get("value"),
                        "timestamp_utc": m.get("period", {}).get("datetimeTo", {}).get("utc"),
                    })
            except Exception as exc:
                logger.warning("Skipping sensor %d (%s): %s", sensor["id"], param, exc)

    logger.info("Total raw records fetched: %d", len(records))
    return records


# ── 3. clean_values ───────────────────────────────────────────

def clean_values(df):
    return df.filter(
        F.col("value").isNotNull() &
        (F.col("value") > 0) &
        (F.col("value") < 10000) &
        F.col("timestamp_utc").isNotNull()
    )


# ── 4. pivot_wide ─────────────────────────────────────────────

def pivot_wide(df):
    df = df.withColumn(
        "timestamp_utc",
        F.to_timestamp(F.col("timestamp_utc"), "yyyy-MM-dd'T'HH:mm:ss'Z'"),
    )
    return (
        df.groupBy("location_id", "location_name", "city", "timestamp_utc")
          .pivot("parameter", POLLUTANTS)
          .agg(F.avg("value"))
          .fillna(-1.0)
    )


# ── 5. compute_aqi_label ──────────────────────────────────────

def compute_aqi_label(df):
    pm = F.coalesce(
        F.when(F.col("pm25") > 0, F.col("pm25")),
        F.when(F.col("pm10") > 0, F.col("pm10") * 0.6),
    )
    return df.withColumn(
        "aqi_category",
        F.when(pm.isNull(), None)   # <-- drop rows with no PM data
         .when(pm <= 30,  0)
         .when(pm <= 60,  1)
         .when(pm <= 90,  2)
         .when(pm <= 120, 3)
         .when(pm <= 250, 4)
         .otherwise(5)
         .cast(IntegerType()),
    ).filter(F.col("aqi_category").isNotNull())  # drop null AQI rows


# ── 6. write_parquet ──────────────────────────────────────────

def write_parquet(df, output_path: str) -> None:
    count = df.count()
    logger.info("Writing %d rows to %s", count, output_path)
    df = df.withColumn(
        "ingestion_date",
        F.lit(datetime.now(timezone.utc).strftime("%Y-%m-%d")),
    )
    (
        df.write
          .mode("overwrite")
          .partitionBy("ingestion_date")
          .option("compression", "snappy")
          .parquet(output_path.rstrip("/"))
    )
    logger.info("Write complete.")


# ── Run ───────────────────────────────────────────────────────

api_key = get_api_key(args["SECRET_NAME"])
records = fetch_from_openaq(api_key, lookback_days=int(args["LOOKBACK_DAYS"]))

if not records:
    logger.error("No records fetched. Writing empty result and exiting.")
    job.commit()
    sys.exit(0)

df = spark.createDataFrame(
    spark.sparkContext.parallelize(records, 8),
    schema=RAW_SCHEMA,
)

df = clean_values(df)
df = pivot_wide(df)
df = compute_aqi_label(df)

write_parquet(df, args["S3_OUTPUT_PATH"])

job.commit()
logger.info("Glue job complete.")