import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from components.query_handler import search_and_generate_response
from components.upload_interface import render_upload_ui
from components.rewrite_interface import render_rewrite_ui  # ✅ NEW IMPORT

# Set page configuration
st.set_page_config(page_title="🎥 BnB RAG System", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section:", [
    "💬 Query Assistant",
    "📤 Upload Content",
    "✏️ Rewrite Blog"  # ✅ NEW OPTION
])

# Main: Query Assistant
if page == "💬 Query Assistant":
    st.title("💬 Ask the RAG Assistant")
    st.markdown("Use this assistant to generate content based on your BnB blogs, brand voice, and style inspiration.")

    user_query = st.text_input("Ask a question:", placeholder="e.g., How can we improve B2B video marketing?")
    if user_query and st.button("Generate Response"):
        with st.spinner("Generating your response..."):
            result = search_and_generate_response(user_query)

        st.subheader("Response")
        st.markdown(result["response"])

        st.divider()
        with st.expander("📂 View Matched Sources"):
            st.write("**OUR Blogs**")
            for doc, score in result["matches"]["our_blogs"]:
                st.markdown(f"- `{doc}` (Score: {score:.4f})")

            st.write("**STYLE Blogs**")
            for doc, score in result["matches"]["blog_styles"]:
                st.markdown(f"- `{doc}` (Score: {score:.4f})")

            st.write("**Brand Documents**")
            for doc, score in result["matches"]["docs"]:
                st.markdown(f"- `{doc}` (Score: {score:.4f})")

# Main: Upload Mode
elif page == "📤 Upload Content":
    render_upload_ui()

# Main: Rewrite Blog
elif page == "✏️ Rewrite Blog":
    render_rewrite_ui()
