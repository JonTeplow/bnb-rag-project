# Re-import everything after state reset
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

# Initialize boto3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Files to delete (the wrongly named ones)
files_to_delete = [
    "vector_store/blogs_our_.index",
    "vector_store/metadata_blogs_our_.pkl",
    "vector_store/blogs_style_.index",
    "vector_store/metadata_blogs_style_.pkl",
    "vector_store/docs_.index",
    "vector_store/metadata_docs_.pkl"
]

# Delete files
deleted = []
for key in files_to_delete:
    try:
        s3_client.delete_object(Bucket=S3_BUCKET, Key=key)
        deleted.append(key)
    except Exception as e:
        print(f"❌ Failed to delete {key}: {e}")

