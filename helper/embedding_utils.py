import os
import pickle
import faiss
import boto3
import numpy as np
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize S3 and embedding model
s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Map prefixes to standardized names
FAISS_KEY_MAP = {
    "blogs/our/": "our_blogs",
    "blogs/style/": "blog_styles",
    "docs/": "docs"
}

def update_embeddings_after_upload(prefix):
    if prefix not in FAISS_KEY_MAP:
        raise ValueError(f"Unknown prefix '{prefix}' – please add it to FAISS_KEY_MAP.")

    index, metadata = load_index_and_metadata(prefix)
    new_docs = fetch_new_documents_from_s3(prefix, metadata)

    if not new_docs:
        print("No new documents to index.")
        return

    new_embeddings = [embedding_model.embed_query(doc['text']) for doc in new_docs]
    new_embeddings = np.array(new_embeddings, dtype="float32")

    index.add(new_embeddings)
    metadata.extend(new_docs)

    save_index_and_metadata(prefix, index, metadata)
    print(f"✅ Updated embeddings for {len(new_docs)} new documents.")

def load_index_and_metadata(prefix):
    key = FAISS_KEY_MAP[prefix]
    index_path = f"vector_store/faiss_index_{key}"
    metadata_path = f"vector_store/metadata_{key}.pkl"

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    EMBEDDING_DIMENSION = 1536
    return faiss.IndexFlatL2(EMBEDDING_DIMENSION), []

def save_index_and_metadata(prefix, index, metadata):
    key = FAISS_KEY_MAP[prefix]
    local_index_path = f"vector_store/faiss_index_{key}"
    local_metadata_path = f"vector_store/metadata_{key}.pkl"

    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, local_index_path)
    with open(local_metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    with open(local_index_path, "rb") as f:
        s3_client.put_object(Bucket=S3_BUCKET, Key=f"vector_store/faiss_index_{key}", Body=f.read())
    with open(local_metadata_path, "rb") as f:
        s3_client.put_object(Bucket=S3_BUCKET, Key=f"vector_store/metadata_{key}.pkl", Body=f.read())

def fetch_new_documents_from_s3(prefix, existing_metadata):
    existing_keys = {meta['s3_key'] for meta in existing_metadata}
    new_docs = []

    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'] not in existing_keys:
                    new_docs.append({
                        's3_key': obj['Key'],
                        'text': download_and_read_s3_object(obj['Key'])
                    })
    except ClientError as e:
        print(f"❌ Failed to list objects in S3 with prefix {prefix}: {e}")

    return new_docs

def download_and_read_s3_object(s3_key):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return response['Body'].read().decode("utf-8")
    except ClientError as e:
        print(f"❌ Failed to download {s3_key} from S3: {e}")
        return ""
