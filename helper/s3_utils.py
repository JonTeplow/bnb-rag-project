import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
import streamlit as st

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=st.secrets("AWS_SECRET_ACCESS_KEY")
)

def upload_to_s3(file_content, s3_key, bucket_name=os.getenv("S3_BUCKET")):
    """
    Uploads a file to an S3 bucket.
    :param file_content: Content to upload.
    :param s3_key: S3 object key.
    :param bucket_name: Name of the S3 bucket.
    :return: True if upload was successful, else False.
    """
    try:
        s3_client.put_object(Body=file_content, Bucket=bucket_name, Key=s3_key)
        print(f"✅ Uploaded {s3_key} to {bucket_name}")
        return True
    except NoCredentialsError:
        print("❌ AWS credentials not available.")
    except ClientError as e:
        print(f"❌ Failed to upload {s3_key} to {bucket_name}: {e}")
    return False

def file_exists_in_s3(prefix, filename, bucket_name=os.getenv("S3_BUCKET")):
    """
    Checks if a file exists in an S3 bucket under a specific prefix.
    :param prefix: S3 prefix (folder path).
    :param filename: Name of the file to check.
    :param bucket_name: Name of the S3 bucket.
    :return: True if file exists, else False.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'] == f"{prefix}{filename}":
                    return True
        return False
    except ClientError as e:
        print(f"❌ Failed to list objects in {bucket_name}/{prefix}: {e}")
        return False
