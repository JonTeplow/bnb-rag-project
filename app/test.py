import os
import faiss
import numpy as np
import pickle
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing! Add it to your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Paths
VECTOR_DB_DIR = "data/vector_store/"
OUR_BLOGS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_our_blogs")
STYLE_BLOGS_INDEX = os.path.join(VECTOR_DB_DIR, "faiss_index_blog_styles")
OUR_BLOGS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_our_blogs.pkl")
STYLE_BLOGS_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_blog_styles.pkl")
OUR_BLOGS_TEXTS = "data/blogs_our/"
STYLE_BLOGS_TEXTS = "data/blogs_style/"

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

def load_faiss_index(index_path, metadata_path):
    """Load FAISS index and metadata."""
    print(f"✅ Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    print(f"✅ Loading metadata from {metadata_path}...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✅ Successfully loaded {len(metadata)} metadata entries.")
    return index, metadata

# Load indices and metadata
our_blogs_index, our_blogs_metadata = load_faiss_index(OUR_BLOGS_INDEX, OUR_BLOGS_METADATA)
style_blogs_index, style_blogs_metadata = load_faiss_index(STYLE_BLOGS_INDEX, STYLE_BLOGS_METADATA)

def load_text(file_name, directory):
    """Load text content from a file."""
    file_path = os.path.join(directory, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Error reading {file_name}: {e}")
        return ""

def search_in_indices(query, top_k=5):
    """Search FAISS indices for content (OUR BLOGS) and style (STYLE BLOGS)."""
    print(f"🔍 Searching for: {query}")
    query_embedding = embedding_model.embed_query(query)
    
    # Search in OUR BLOGS for content
    num_our_blogs = max(1, int(top_k * 0.8))
    distances_our, indices_our = our_blogs_index.search(np.array([query_embedding], dtype="float32"), num_our_blogs)
    
    # Search in STYLE BLOGS for tone reference
    num_style_blogs = max(1, top_k - num_our_blogs)
    distances_style, indices_style = style_blogs_index.search(np.array([query_embedding], dtype="float32"), num_style_blogs)
    
    matched_our_blogs = [(our_blogs_metadata[i], load_text(our_blogs_metadata[i], OUR_BLOGS_TEXTS)) for i in indices_our[0] if i < len(our_blogs_metadata)]
    matched_style_blogs = [(style_blogs_metadata[i], load_text(style_blogs_metadata[i], STYLE_BLOGS_TEXTS)) for i in indices_style[0] if i < len(style_blogs_metadata)]
    
    return matched_our_blogs, matched_style_blogs

def generate_content_suggestions(query):
    """Generate content using OUR BLOGS for content and STYLE BLOGS for tone/format."""
    matched_our, matched_style = search_in_indices(query)
    
    # Extract longer text snippets from OUR BLOGS (at least 2000 characters per snippet)
    our_content = "\n\n".join(f"Content from: {title}\n{text[:2000]}..." for title, text in matched_our if text)
    
    # Extract structured style references from STYLE BLOGS (to match format & tone)
    style_examples = "\n\n".join(f"Style Reference: {title}\n{text[:2000]}..." for title, text in matched_style if text)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a content strategist. Your job is to rewrite content from OUR BLOGS while mimicking the tone and structure of STYLE BLOGS. Strictly follow the format and style of STYLE BLOGS without adding extra commentary. Do not introduce new examples, brands, case studies, or insights unless they exist in the retrieved OUR BLOGS content. If a concept, brand, or company is not present in OUR BLOGS, omit it entirely. Do not summarize or add conclusions—only rewrite the retrieved content."},
            {"role": "user", "content": f"Content Source (OUR BLOGS): \n{our_content}\n\nStyle Reference (STYLE BLOGS): \n{style_examples}\n\nRewrite the content from OUR BLOGS using the exact structure, formatting, and tone of STYLE BLOGS. Do not add extra details, introduce new examples, or create additional insights. Ensure that every sentence is based strictly on the retrieved content from OUR BLOGS. Do not summarize or add conclusions."}
        ],
        max_tokens=800
    )
    
    return response.choices[0].message.content

def main():
    """Main function to run content generation script."""
    query = input("Enter your content query: ")
    content_suggestions = generate_content_suggestions(query)
    
    print("\n✨ Content Suggestions:")
    print(content_suggestions)

if __name__ == "__main__":
    main()
