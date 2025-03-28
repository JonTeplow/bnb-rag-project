import os
import faiss
import numpy as np
import pickle
import boto3
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = "bnbcontentstudio"

# FAISS and Embeddings Configuration
VECTOR_DB_DIR = "data/vector_store/"
OUR_BLOGS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_our_blogs")
STYLE_BLOGS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_blog_styles")
DOCS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_docs")

OUR_BLOGS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_our_blogs.pkl")
STYLE_BLOGS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_blog_styles.pkl")
DOCS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_docs.pkl")

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def list_s3_files(prefix):
    """List all files in a given S3 prefix."""
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def download_s3_file(s3_key, local_path):
    """Download a file from S3."""
    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)

def load_existing_metadata(metadata_path):
    """Load metadata if exists, otherwise return an empty list."""
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    return []

def generate_embeddings_and_update_faiss(s3_prefix, index_path, metadata_path):
    """Fetch new/updated files from S3, generate embeddings, and update FAISS."""
    print(f"🔄 Checking for new content in {s3_prefix}...")
    
    existing_files = set(load_existing_metadata(metadata_path))
    s3_files = list_s3_files(s3_prefix)
    new_files = [f for f in s3_files if f not in existing_files]
    
    if not new_files:
        print(f"✅ No new updates found for {s3_prefix}")
        return
    
    print(f"🆕 Found {len(new_files)} new or updated files in {s3_prefix}!")
    
    texts, file_names = [], []
    for s3_file in new_files:
        local_file = os.path.join("temp", os.path.basename(s3_file))
        download_s3_file(s3_file, local_file)
        
        with open(local_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                texts.append(text)
                file_names.append(s3_file)
    
    if texts:
        embeddings = embedding_model.embed_documents(texts)
        embedding_matrix = np.array(embeddings, dtype="float32")
        
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        
        index.add(embedding_matrix)
        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(existing_files.union(set(new_files)), f)
        
        print(f"✅ Updated FAISS index for {s3_prefix} with {len(new_files)} new embeddings!")

# Process all categories
generate_embeddings_and_update_faiss("blogs/our/", OUR_BLOGS_INDEX, OUR_BLOGS_METADATA)
generate_embeddings_and_update_faiss("blogs/style/", STYLE_BLOGS_INDEX, STYLE_BLOGS_METADATA)
generate_embeddings_and_update_faiss("docs/", DOCS_INDEX, DOCS_METADATA)

print("🚀 All embeddings updated!")
