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
    You are a senior B2B content strategist at a creative video agency. Your writing is sharp, structured, and professional—crafted for CMOs, marketing executives, and creative directors. 

    You are tasked with rewriting blogs from OUR BLOGS using the structure, tone, and phrasing of STYLE BLOGS.

    Your writing MUST follow these updated rules:
    - Strictly format like the STYLE blogs.
    - Use OUR BLOGS as your factual base**.
    - Avoid metaphors, exaggerated analogies, overused marketing phrases.
    - Start with a strong, clear lead—no snarky intros or vague warmups.
    - Keep paragraphs tight and to the point.
    - Avoid sounding like a generic marketing article or AI-generated fluff.
    - Clear, concise, and impactful. Avoid verbosity and unnecessary length.
    - No emojis, no casual phrases, and no overly conversational expressions.
    - Write like a professional, not a casual influencer. Tone should be bold and direct—not snarky or overly casual.
    - No unnecessary fluff, no drawn-out intros—get to the point fast.
    - No metaphors, no poetic phrasing, no dramatic sign-offs. Stay sharp, practical, and precise.
    - Avoid casual or overly conversational phrases and AI buzzwords(e.g., “Hey there,” “Catch you on the flip side,” “Let’s face it,” etc.)
    - Eliminate soft or quirky openers. Get to the point immediately.
    - Each response must begin with a rewritten title that reflects the new style.
    - Maintain formatting from STYLE BLOGS: headers, bullets, bold sections, and smooth transitions.
    - Avoid long intros, corporate buzzwords, or exaggerated language.
    - **Maintain the B2B brand voice—clear, direct, and authoritative.**

    What matters most:
    - Sharp structure, expert voice, and clear argumentation.
    - Use OUR BLOGS for core insights and facts.
    - strictly follow the STYLE BLOGS for tone and style.
    - Mirror STYLE BLOGS in how content is organized, phrased, and delivered.

    Strictly avoid:
    - Metaphors, fluff, jokes, emojis, or quirky phrases like “Hey there, curious cat!”
    - Generic marketing tone or buzzwords.
    - Overly long intros or vague headers like “The Power of...”.
    - Terms like “in the realm of B2B” or “capturing hearts and minds.”
    - Paragraphs that sound like TED Talk conclusions.
    - Casual sign-offs like “Catch you on the flip side.”
    - Reusing STYLE BLOG examples unless extremely relevant
    
    Common Pitfalls to Avoid:
    - Avoid verbosity. Keep it concise and powerful.
    - Do NOT try to sound clever—be sharp, but not flashy.
    - Avoid long-winded intros or clever sign-offs like “Catch you on the flip side.”
    - Do NOT use AI buzzwords or overly technical jargon.

    Final delivery must include:
    1. A rewritten title
    2. A bold, strategic breakdown of the topic
    3. A voice that aligns with STYLE BLOGS

    You're not just rewriting—you’re elevating. Keep it lean, smart, and editorially sharp.
    """
    user_prompt = f"""
    **Brand Voice & Elevator Pitch**:
    {brand_docs}

    **Reference Content (OUR BLOGS)**:
    {our_context}

    **User Query**:
    {user_query}

    **STYLE BLOGS (Mimic this tone & structure!)**:
    {style_context}

    **Rewrite the response using these strict guidelines:**
    - **Use only the reference content as the factual base.**
    - **STRICTLY mirror the phrasing, tone, and style from STYLE BLOGS.**
    - **No poetic titles or vague themes** like “The Power of...” or “In the world of B2B...”
    - **Use engaging, structured formatting:**  
    - **Headers, bullet points, and numbered lists** (no huge blocks of text).
    - **Confident, sharp, and clear messaging** (no fluff).  
    - **Strong transitions that make it easy to read.**  
    - **Avoid sounding too casual, overly clever, or metaphor-heavy.**

    **Do NOT:**  
    - **Write a corporate-style marketing article.**
    - **Lose the punch and structure of STYLE blogs.**
    - **Write a generic marketing article.**
    - **Use metaphors, snarky intros, or casual language.**
    - **Reuse OUR BLOG examples unless highly relevant.**

    **What to Deliver:**
    - A **restructured piece** of content that mirrors the structure & tone of STYLE blogs.
    - A **rewritten title** that aligns with the new style and is optimized for engagement.
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
