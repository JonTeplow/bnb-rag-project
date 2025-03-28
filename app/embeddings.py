import os
import faiss
import numpy as np
import pickle
import boto3
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()

# Get API credentials from .env
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

# S3 Directories
OUR_BLOGS_PREFIX = "blogs/our/"
STYLE_BLOGS_PREFIX = "blogs/style/"
DOCS_PREFIX = "docs/"
VECTOR_DB_DIR = "vector_store/"

# FAISS Index Paths
OUR_BLOGS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_our_blogs.index")
STYLE_BLOGS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_blog_styles.index")
DOCS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_docs.index")

OUR_BLOGS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_our_blogs.pkl")
STYLE_BLOGS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_blog_styles.pkl")
DOCS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_docs.pkl")

# Initialize S3 Client
s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

def list_s3_files(prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def fetch_text_from_s3(s3_key):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return response["Body"].read().decode("utf-8")
    except Exception as e:
        print(f"❌ Error fetching {s3_key}: {e}")
        return None

def save_faiss_index(index, local_path, s3_key):
    try:
        # Save FAISS index locally first
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        faiss.write_index(index, local_path)

        # Upload the local file to S3
        with open(local_path, "rb") as f:
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f.read())

        print(f"✅ FAISS Index saved to S3: {s3_key}")
    except Exception as e:
        print(f"❌ Error saving FAISS index to S3: {e}")

def save_metadata(metadata, s3_key):
    buffer = BytesIO()
    pickle.dump(metadata, buffer)
    buffer.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue())

def load_existing_embeddings(s3_key):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        temp_path = "temp_faiss.index"
        with open(temp_path, "wb") as f:
            f.write(response["Body"].read())
        return faiss.read_index(temp_path)
    except Exception:
        return None

def generate_and_save_embeddings(prefix, index_path, index_s3_key, metadata_path):
    existing_files = []
    index = load_existing_embeddings(index_s3_key) or faiss.IndexFlatL2(1536)

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=metadata_path)
        existing_files = pickle.loads(response["Body"].read())
    except Exception:
        pass

    new_files = list_s3_files(prefix)
    new_files = [f for f in new_files if f not in existing_files]

    if not new_files:
        print(f"✅ No new files to embed for {prefix}.")
        return

    print(f"🔄 Generating embeddings for {len(new_files)} new files in {prefix}...")

    texts, file_names = [], []
    for s3_key in new_files:
        text = fetch_text_from_s3(s3_key)
        if text:
            texts.append(text)
            file_names.append(s3_key)

    if texts:
        embeddings = embedding_model.embed_documents(texts)
        embedding_matrix = np.array(embeddings, dtype="float32")
        index.add(embedding_matrix)

        save_faiss_index(index, index_path, index_s3_key)
        save_metadata(existing_files + new_files, metadata_path)

        print(f"✅ Embeddings stored for {prefix} in FAISS & S3.")
    else:
        print(f"⚠️ No valid text data found in {prefix}.")

# Run embeddings update for all content types
generate_and_save_embeddings(OUR_BLOGS_PREFIX, OUR_BLOGS_INDEX, "vector_store/faiss_index_our_blogs", OUR_BLOGS_METADATA)
generate_and_save_embeddings(STYLE_BLOGS_PREFIX, STYLE_BLOGS_INDEX, "vector_store/faiss_index_blog_styles", STYLE_BLOGS_METADATA)
generate_and_save_embeddings(DOCS_PREFIX, DOCS_INDEX, "vector_store/faiss_index_docs", DOCS_METADATA)

print("🚀 Embeddings update complete!")
