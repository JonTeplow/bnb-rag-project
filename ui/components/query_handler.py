import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from helper.vector_store_utils import (
    load_all_indices_and_metadata,
    search_faiss_index,
    load_text_from_s3
)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing in your .env")

# Initialize OpenAI client
openai_client = OpenAI(api_key=api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Truncation helper
def truncate_text(text, max_tokens=1500):
    words = text.split()
    return " ".join(words[:max_tokens])

def search_and_generate_response(user_query, k=5):
    """
    Full RAG pipeline:
    1. Search 'our_blogs', 'blog_styles', and 'docs'
    2. Load matched content from S3
    3. Use OpenAI to generate a final response
    """
    indices, metadata = load_all_indices_and_metadata()
    print("✅ Loaded FAISS Indices:", indices.keys())  # Debugging
    print("✅ Loaded FAISS Metadata:", metadata.keys())  # Debugging

    # Search top-k matches in each index
    our_results = search_faiss_index(user_query, indices["our_blogs"], metadata["our_blogs"], k)
    style_results = search_faiss_index(user_query, indices["blog_styles"], metadata["blog_styles"], k)
    docs_results = search_faiss_index(user_query, indices["docs"], metadata["docs"], 2)

    # Load matched content
    def fetch_text(matches):
        return "\n\n".join([
            load_text_from_s3(doc_key)
            for doc_key, _ in matches
        ])

    our_context = truncate_text(fetch_text(our_results), max_tokens=1000)
    style_context = truncate_text(fetch_text(style_results), max_tokens=800)
    brand_docs = truncate_text(fetch_text(docs_results), max_tokens=600)

    # Compose prompt
    system_prompt = """
    You are a B2B content strategist at a creative video agency. Your writing is authoritative, insightful, and professional—geared toward marketing executives and creative directors.  
    You rewrite content from OUR BLOGS using the deep analysis, tone, and structure found in STYLE BLOGS.   
    Your writing must **STRICTLY** match the **tone, phrasing, and energy of STYLE BLOGS** while using OUR BLOGS as your factual base.

    🚀 **Your writing style MUST follow these rules:**
    - **Format like the STYLE blogs** (snappy, clear, engaging).  
    - **Use OUR BLOGS as your factual base** (deep analysis, data-driven insights)**
    - **No numbered steps or listicle-style formatting.** Instead, structure content like an **expert industry breakdown.**  
    - **No unnecessary fluff, no drawn-out intros**—get to the point fast.
    - **No emojis, no casual phrases, no unnecessary emphasis.** (E.g., "Let’s face it," "The game has changed," etc.)    
    - **Write in a bold, confident, and modern tone** that’s easy to read.  
    - **Use case studies and logical argumentation.** STYLE blogs focus on **analysis, trends, and expert-level storytelling.**   
    - **Maintain a professional but engaging voice**—informative yet dynamic.  
    - **Maintain the B2B brand voice—clear, direct, and strategic.** 

    🚨 **DO NOT:**  
    - **Sound like a generic marketing article.**  
    - **Lose the punchy, structured rhythm of STYLE blogs.**  
    - **Overload with long paragraphs or corporate jargon.**
    - **use AI buzzwords or overly technical terms.**

    💡 **Your mission:**  
    Rewrite content using the **exact same structure & style as the STYLE blogs** while keeping the facts accurate from OUR blogs.  
    """

    user_prompt = f"""
    📌 **Brand Voice & Elevator Pitch**:
    {brand_docs}

    📌 **Reference Content (OUR BLOGS)**:
    {our_context}

    📌 **User Query**:
    {user_query}

    📌 **STYLE BLOGS (Mimic this tone & structure!)**:
    {style_context}

    🎯 **Rewrite the response using these strict guidelines:**
    - **Use only the reference content as the factual base.**
    - **STRICTLY mirror the phrasing, tone, and style from STYLE BLOGS.**
    - **Use engaging, structured formatting:**  
    - **Headers, bullet points, and numbered lists** (no huge blocks of text).
    - **Confident, sharp, and clear messaging** (no fluff).  
    - **Strong transitions that make it easy to read.**  

    🛑 **Do NOT:**  
    - **Write a corporate-style marketing article.**
    - **Lose the punch and structure of STYLE blogs.**
    - **Use the same examples from OUR blogs unless highly relevant.**

    📢 **Final step:**  
    Deliver a **sharp and engaging** response that blends strategic storytelling with practical insights.
    """


    # Generate OpenAI response
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )

    return {
        "response": response.choices[0].message.content.strip(),
        "matches": {
            "our_blogs": our_results,
            "blog_styles": style_results,
            "docs": docs_results
        }
    }
