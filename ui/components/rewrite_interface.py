import os
import streamlit as st
from io import BytesIO
from datetime import datetime
from docx import Document

from helper.docx_utils import extract_text_from_docx
from helper.vector_store_utils import load_all_indices_and_metadata, search_faiss_index, load_text_from_s3
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# Initialize embedding model and OpenAI client
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_blog_with_style(original_text):
    indices, metadata = load_all_indices_and_metadata()

    # Retrieve context from style blogs and brand docs
    dummy_query = "Rewrite this blog using our brand tone and style"
    style_matches = search_faiss_index(dummy_query, indices["blog_styles"], metadata["blog_styles"], k=2)
    brand_matches = search_faiss_index(dummy_query, indices["docs"], metadata["docs"], k=2)

    def fetch_text(matches):
        return "\n\n".join([load_text_from_s3(doc_key) for doc_key, _ in matches])

    style_context = fetch_text(style_matches)
    brand_context = fetch_text(brand_matches)

    # Construct prompt
    system_prompt = f"""You are a senior B2B content strategist at a creative video agency.
Use the following blog as the base content. You must rewrite it strictly using the tone and style from the STYLE BLOGS.
Maintain the tone and personality from the Brand Voice and Elevator Pitch.
The result should be engaging, direct, creative, and human. Do not be robotic or overly polished."""

    user_prompt = f"""
Brand Voice & Elevator Pitch:
{brand_context}

Style Guide Examples:
{style_context}

Original Blog Content:
{original_text}

Rewrite this blog in the exact tone and style of the STYLE BLOGS.

Make it bold, creative, and slightly sassy. Use storytelling, metaphors, and personality. Make sure the final blog grabs attention, avoids corporate fluff, and sounds like it came from a top-tier creative strategist.
"""

    # Generate the rewritten version
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content


def render_rewrite_ui():
    st.title("Rewrite an Existing Blog")
    st.markdown("""
Use this interface to rewrite an existing blog using your brand voice and the tone/style of the reference blogs.
You can paste the blog directly or upload a Word document (.docx).
""")

    option = st.radio("Choose input method:", ["Paste blog text", "Upload .docx file"], horizontal=True)
    blog_text = ""

    if option == "Paste blog text":
        blog_text = st.text_area("Paste your blog content below:", height=300)
    else:
        uploaded_doc = st.file_uploader("Upload a .docx file", type=["docx"])
        if uploaded_doc:
            blog_text = extract_text_from_docx(uploaded_doc)

    if blog_text.strip():
        if st.button("Rewrite Blog"):
            with st.spinner("Rewriting using style and tone reference..."):
                rewritten_text = rewrite_blog_with_style(blog_text)

                st.subheader("Rewritten Blog")
                st.write(rewritten_text)

                # Prepare downloadable .docx file
                docx_file = BytesIO()
                doc = Document()
                doc.add_paragraph(rewritten_text)
                doc.save(docx_file)
                docx_file.seek(0)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download as .docx",
                    data=docx_file,
                    file_name=f"rewritten_blog_{timestamp}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    else:
        st.info("Paste a blog or upload a document to begin.")
