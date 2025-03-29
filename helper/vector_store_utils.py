import boto3
import faiss
import pickle
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
import os
import tempfile
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Load environment variables
load_dotenv()
# AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# S3_BUCKET = os.getenv("S3_BUCKET")
# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET = st.secrets["S3_BUCKET"]

# Initialize S3 client and embedding model
s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# S3 keys for FAISS index and metadata
VECTOR_INDEX_KEYS = {
    "our_blogs": "vector_store/faiss_index_our_blogs",
    "blog_styles": "vector_store/faiss_index_blog_styles",
    "docs": "vector_store/faiss_index_docs"
}
METADATA_KEYS = {
    "our_blogs": "vector_store/metadata_our_blogs.pkl",
    "blog_styles": "vector_store/metadata_blog_styles.pkl",
    "docs": "vector_store/metadata_docs.pkl"
}
import os

def debug_faiss_files():
    vector_store_path = "vector_store/"
    if os.path.exists(vector_store_path):
        print("✅ FAISS folder exists")
        print("📂 Files in vector store:", os.listdir(vector_store_path))
    else:
        print("❌ FAISS folder is missing!")

debug_faiss_files()  # Run this in Streamlit to see FAISS files
def load_text_from_s3(s3_key):
    """Fetch and decode text content from an S3 object (e.g., brand voice, elevator pitch)."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return response["Body"].read().decode("utf-8").strip()
    except Exception as e:
        print(f"❌ Failed to load text from {s3_key}: {e}")
        return ""
    
def load_faiss_index_from_s3(s3_client, bucket, s3_key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        index_bytes = response["Body"].read()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(index_bytes)
            temp_path = temp_file.name

        index = faiss.read_index(temp_path)
        print(f"✅ Loaded FAISS index from {s3_key}")
        return index

    except Exception as e:
        print(f"❌ Failed to load {s3_key} from S3: {e}")
        return None

def load_metadata_from_s3(s3_key):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        metadata = pickle.loads(response["Body"].read())
        print(f"✅ Loaded metadata for {len(metadata)} documents from {s3_key}")
        return metadata
    except Exception as e:
        print(f"❌ Failed to load metadata from {s3_key}: {e}")
        return []

def search_faiss_index(query_text, index, metadata, k=5):
    try:
        query_embedding = np.array(embedding_model.embed_query(query_text), dtype="float32").reshape(1, -1)
        distances, indices_found = index.search(query_embedding, k)
        results = []
        for i in range(len(indices_found[0])):
            doc_index = indices_found[0][i]
            if doc_index < len(metadata):
                results.append((metadata[doc_index], distances[0][i]))
        return results
    except Exception as e:
        print(f"❌ Error during FAISS search: {e}")
        return []

def load_all_indices_and_metadata():
    indices = {}
    metadata = {}

    for key, s3_key in VECTOR_INDEX_KEYS.items():
        index = load_faiss_index_from_s3(s3_client, S3_BUCKET, s3_key)
        if index:
            indices[key] = index

    for key, s3_key in METADATA_KEYS.items():
        meta = load_metadata_from_s3(s3_key)
        metadata[key] = meta

    return indices, metadata
