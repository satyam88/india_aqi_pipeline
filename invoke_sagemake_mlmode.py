import boto3
import json

client = boto3.client("sagemaker-runtime", region_name="ap-south-1")

response = client.invoke_endpoint(
    EndpointName="india-aqi-endpoint",
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps({
        "location": "Delhi",
        "datetime": "2024-12-10T08:00:00"
    }),
)

result = json.loads(response["Body"].read())
print(result)