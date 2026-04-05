"""
invoke_endpoint.py
==================
Test the deployed India AQI endpoint.

Usage:
  # Minimal (time-based prediction only, pollutants default to -1 sentinel):
  python invoke_endpoint.py --location Delhi --datetime "2026-03-10T08:00:00"

  # With known pollutant readings:
  python invoke_endpoint.py --location Mumbai --datetime "2026-03-10T08:00:00" \
      --pm25 85.2 --pm10 120.0 --no2 45.0

  # Raw numeric features (legacy):
  python invoke_endpoint.py --raw "[0.5, 1.2, -1, -1, -1, -1, 8, 3, 1, 0, ...]"
"""

import argparse
import json
import logging

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ENDPOINT_NAME = "india-aqi-endpoint"
REGION        = "ap-south-1"


def parse_args():
    p = argparse.ArgumentParser(description="Invoke India AQI SageMaker endpoint")
    p.add_argument("--endpoint",  type=str, default=ENDPOINT_NAME)
    p.add_argument("--region",    type=str, default=REGION)
    # High-level payload
    p.add_argument("--location",  type=str, default=None, help="City name, e.g. Delhi")
    p.add_argument("--datetime",  type=str, default=None, help="ISO datetime, e.g. 2026-03-10T08:00:00")
    p.add_argument("--pm25",      type=float, default=None)
    p.add_argument("--pm10",      type=float, default=None)
    p.add_argument("--no2",       type=float, default=None)
    p.add_argument("--so2",       type=float, default=None)
    p.add_argument("--co",        type=float, default=None)
    p.add_argument("--o3",        type=float, default=None)
    # Legacy raw payload
    p.add_argument("--raw",       type=str,   default=None,
                   help="JSON list of raw numeric features, e.g. '[[0.5, 1.2, ...]]'")
    return p.parse_args()


def build_payload(args) -> dict:
    if args.raw:
        instances = json.loads(args.raw)
        if not isinstance(instances[0], list):
            instances = [instances]
        return {"instances": instances}

    if not args.location or not args.datetime:
        raise ValueError("Provide --location and --datetime, or --raw for legacy mode.")

    payload = {
        "location": args.location,
        "datetime": args.datetime,
    }
    for field in ("pm25", "pm10", "no2", "so2", "co", "o3"):
        val = getattr(args, field)
        if val is not None:
            payload[field] = val

    return payload


def invoke(endpoint: str, region: str, payload: dict) -> dict:
    client = boto3.client("sagemaker-runtime", region_name=region)
    logger.info("Invoking endpoint: %s", endpoint)
    logger.info("Payload: %s", json.dumps(payload, indent=2))

    response = client.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read())


def main():
    args    = parse_args()
    payload = build_payload(args)
    result  = invoke(args.endpoint, args.region, payload)

    print("\n── Prediction Result ─────────────────────────────────────")
    print(f"  AQI Category : {result.get('aqi_labels', ['?'])[0]}")
    print(f"  Class Index  : {result.get('predictions', ['?'])[0]}")
    print(f"  Class Order  : {result.get('class_order')}")
    print("\n  Probabilities:")
    probs       = result.get("probabilities", [[]])[0]
    class_order = result.get("class_order", [])
    for label, prob in zip(class_order, probs):
        bar = "█" * int(prob * 30)
        print(f"    {label:<15} {prob:.4f}  {bar}")
    print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()