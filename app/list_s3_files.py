import boto3
import os
from dotenv import load_dotenv

# Load AWS credentials
load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

# Initialize S3 client
s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# List all files under vector_store/
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="vector_store/")
if "Contents" in response:
    print("📦 Files in vector_store/:")
    for obj in response["Contents"]:
        print(" -", obj["Key"])
else:
    print("❌ No files found in vector_store/")
