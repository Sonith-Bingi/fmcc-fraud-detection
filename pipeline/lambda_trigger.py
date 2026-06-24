"""
Lambda function — triggered by S3 PUT event on fmcc-raw-cdrs bucket.
Extracts the S3 key and calls the /process endpoint on the fraud API.
"""
import json
import urllib.request
import urllib.error
import os

API_URL = os.environ["API_GATEWAY_URL"]  # set in Lambda environment variables


def lambda_handler(event, context):
    # Extract S3 key from the event AWS sends
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    s3_key = record["s3"]["object"]["key"]

    print(f"New file detected: s3://{bucket}/{s3_key}")

    # Only process CDR files
    if not s3_key.startswith("cdrs/") or not s3_key.endswith(".csv"):
        print(f"Skipping non-CDR file: {s3_key}")
        return {"statusCode": 200, "body": "skipped"}

    # Call the /process endpoint
    payload = json.dumps({"s3_key": s3_key}).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/process",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=840) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            print(f"Processing complete: {result}")
            return {"statusCode": 200, "body": body}
    except urllib.error.HTTPError as e:
        error = e.read().decode("utf-8")
        print(f"API error {e.code}: {error}")
        raise
