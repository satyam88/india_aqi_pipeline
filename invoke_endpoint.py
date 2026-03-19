import boto3
import json

# Exact 27 columns the model was trained on
COLUMNS = [
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "hour_of_day", "month", "is_rush_hour", "is_weekend",
    "city_Alandur Bus Depot",
    "city_Anand Vihar, New Delhi - DPCC",
    "city_BTM Layout, Bengaluru - KSPCB",
    "city_Collectorate - Gaya - BSPCB",
    "city_Collectorate Jodhpur - RSPCB",
    "city_Delhi Technological University, Delhi - CPCB",
    "city_Haldia, Haldia - WBPCB",
    "city_IHBAS, Delhi - CPCB",
    "city_Income Tax Office, Delhi - CPCB",
    "city_Lalbagh, DN Park",
    "city_MD University, Rohtak - HSPCB",
    "city_Mandir Marg, Delhi - DPCC",
    "city_Peenya, Bengaluru - KSPCB",
    "city_Punjabi Bagh, Delhi - DPCC",
    "city_R K Puram, Delhi - DPCC",
    "city_Vikas Sadan, Gurugram - HSPCB",
    "city_Zoo Park, Hyderabad - TSPCB",
]

AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

def predict(pm25, pm10, no2, so2, co, o3, hour, month, city):
    col_map = {c: i for i, c in enumerate(COLUMNS)}
    row     = [0.0] * len(COLUMNS)

    row[col_map["pm25"]]         = pm25
    row[col_map["pm10"]]         = pm10
    row[col_map["no2"]]          = no2
    row[col_map["so2"]]          = so2
    row[col_map["co"]]           = co
    row[col_map["o3"]]           = o3
    row[col_map["hour_of_day"]]  = hour
    row[col_map["month"]]        = month
    row[col_map["is_rush_hour"]] = 1 if hour in [7,8,9,17,18,19] else 0
    row[col_map["is_weekend"]]   = 0

    city_col = f"city_{city}"
    if city_col in col_map:
        row[col_map[city_col]] = 1.0

    client   = boto3.client("sagemaker-runtime", region_name="ap-south-1")
    response = client.invoke_endpoint(
        EndpointName="india-aqi-endpoint",
        ContentType="text/csv",
        Accept="application/json",
        Body=",".join(str(v) for v in row),
    )
    return json.loads(response["Body"].read())


# Test 1 — Delhi rush hour, high pollution
r1 = predict(85, 120, 40, 15, 1.2, 30, 8, 12,
             "Anand Vihar, New Delhi - DPCC")
print("Delhi rush hour  :", r1["aqi_labels"][0],
      f"({r1['probabilities'][0][r1['predictions'][0]]:.0%} confidence)")

# Test 2 — Bangalore low pollution
r2 = predict(20, 35, 15, 5, 0.5, 45, 10, 6,
             "BTM Layout, Bengaluru - KSPCB")
print("Bangalore morning:", r2["aqi_labels"][0],
      f"({r2['probabilities'][0][r2['predictions'][0]]:.0%} confidence)")

# Test 3 — Hyderabad moderate
r3 = predict(55, 80, 25, 10, 0.8, 35, 18, 3,
             "Zoo Park, Hyderabad - TSPCB")
print("Hyderabad evening:", r3["aqi_labels"][0],
      f"({r3['probabilities'][0][r3['predictions'][0]]:.0%} confidence)")
