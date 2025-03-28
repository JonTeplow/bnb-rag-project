import os
import boto3
import faiss
import pickle
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
import tempfile
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# File paths in S3
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

# Load FAISS index from S3 using a temporary file
def load_faiss_index_from_s3(s3_key):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
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

# Load metadata from S3
def load_metadata_from_s3(s3_key):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        metadata = pickle.loads(response["Body"].read())
        print(f"✅ Loaded metadata for {len(metadata)} documents from {s3_key}")
        return metadata
    except Exception as e:
        print(f"❌ Failed to load metadata from {s3_key}: {e}")
        return []

# Initialize dicts
indices = {}
metadata = {}

# Load all indices and metadata
for key in VECTOR_INDEX_KEYS:
    index = load_faiss_index_from_s3(VECTOR_INDEX_KEYS[key])
    if index:
        indices[key] = index

for key in METADATA_KEYS:
    metadata[key] = load_metadata_from_s3(METADATA_KEYS[key])

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Search Function
def search(query_text, category="our_blogs", k=5):
    if category not in indices:
        raise ValueError(f"❌ Invalid category '{category}'. Choose from: {list(indices.keys())}")

    print(f"\n🔍 Searching '{category}' for: {query_text}")
    query_embedding = np.array(embedding_model.embed_query(query_text), dtype="float32").reshape(1, -1)
    distances, indices_found = indices[category].search(query_embedding, k)

    results = []
    for i in range(len(indices_found[0])):
        doc_index = indices_found[0][i]
        if doc_index < len(metadata[category]):
            results.append((metadata[category][doc_index], distances[0][i]))

    return results

# Example query
query = "How to improve B2B video marketing?"
print("\n🔹 OUR BLOGS - Top Matches:")
for doc, score in search(query, category="our_blogs"):
    print(f"📄 {doc} (Score: {score:.4f})")

print("\n🔹 BLOG STYLES - Top Matches:")
for doc, score in search(query, category="blog_styles"):
    print(f"📄 {doc} (Score: {score:.4f})")

print("\n🔹 DOCS - Top Matches:")
for doc, score in search(query, category="docs"):
    print(f"📄 {doc} (Score: {score:.4f})")
