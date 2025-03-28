import streamlit as st
from helper.s3_utils import upload_to_s3, file_exists_in_s3
from helper.embedding_utils import update_embeddings_after_upload
from helper.scraper import scrape_and_save_blog_text
from helper.docx_utils import extract_text_from_docx
from io import BytesIO
from urllib.parse import urlparse


def render_upload_ui():
    st.title("📤 Upload New Content to RAG System")
    st.markdown("""
    Upload new content or update brand documents. The system will automatically:
    - Store the content on S3
    - Update embeddings
    - Keep the knowledge base in sync ✅
    """)

    st.divider()
    st.header("🗂 Blog Uploads")

    col1, col2 = st.columns(2)

     # --- OUR BLOG ---
    with col1:
        st.subheader("OUR Blog (BnB Reference)")
        with st.form("our_blog_form", clear_on_submit=True):
            our_url = st.text_input("🔗 Paste OUR Blog URL", key="our_blog_url_form")
            confirm_our = st.checkbox("I confirm this is a new OUR blog I want to upload", key="confirm_our_blog")
            submitted_our = st.form_submit_button("Upload OUR Blog")

            if submitted_our:
                if not our_url.strip():
                    st.warning("🚫 Please paste a blog URL.")
                elif not confirm_our:
                    st.warning("🔒 Please confirm the upload before proceeding.")
                else:
                    with st.spinner("📡 Uploading blog to the server..."):
                        filename = urlparse(our_url).path.split("/")[-1] + ".txt"
                        s3_key = f"blogs/our/{filename}"
                        if file_exists_in_s3(s3_key, filename):
                            st.warning("⚠️ This blog already exists in OUR Blogs.")
                        else:
                            success = scrape_and_save_blog_text(our_url, s3_prefix="blogs/our/")
                            if success:
                                update_embeddings_after_upload("blogs/our/")
                                st.success("✅ Blog uploaded and indexed successfully!")


                                

    # --- STYLE BLOG ---
    with col2:
        st.subheader("STYLE Blog (Inspiration Style)")
        with st.form("style_blog_form", clear_on_submit=True):
            style_url = st.text_input("🔗 Paste STYLE Blog URL", key="style_blog_url_form")
            confirm_style = st.checkbox("I confirm this is a new STYLE blog I want to upload", key="confirm_style_blog")
            submitted_style = st.form_submit_button("Upload STYLE Blog")

            if submitted_style:
                if not style_url.strip():
                    st.warning("🚫 Please paste a style blog URL.")
                elif not confirm_style:
                    st.warning("🔒 Please confirm the upload before proceeding.")
                else:
                    with st.spinner("📡 Uploading blog to the server..."):
                        filename = urlparse(style_url).path.split("/")[-1] + ".txt"
                        s3_key = f"blogs/style/{filename}"
                        if file_exists_in_s3(s3_key, filename):
                            st.warning("⚠️ This blog already exists in STYLE Blogs.")
                        else:
                            success = scrape_and_save_blog_text(style_url, s3_prefix="blogs/style/")
                            if success:
                                update_embeddings_after_upload("blogs/style/")
                                st.success("✅ Style blog uploaded and indexed successfully!")


    st.divider()
    st.header("Brand Documents")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Replace Brand Voice (.docx)")
        brand_doc = st.file_uploader("📄 Upload new Brand Voice file", type=["docx"], key="brand_voice")
        if brand_doc and st.button("Replace Brand Voice"):
            text = extract_text_from_docx(BytesIO(brand_doc.read()))
            upload_to_s3(text.encode("utf-8"), "docs/brand_voice.txt")
            update_embeddings_after_upload("docs/")
            st.success("✅ Brand Voice document replaced and embedded!")

    with col4:
        st.subheader("Replace Elevator Pitch (.docx)")
        pitch_doc = st.file_uploader("📄 Upload new Elevator Pitch file", type=["docx"], key="elevator_pitch")
        if pitch_doc and st.button("Replace Elevator Pitch"):
            text = extract_text_from_docx(BytesIO(pitch_doc.read()))
            upload_to_s3(text.encode("utf-8"), "docs/elevator_pitch.txt")
            update_embeddings_after_upload("docs/")
            st.success("✅ Elevator Pitch document replaced and embedded!")

    st.divider()
    st.info("🛡️ All uploads are automatically checked for duplicates and embedded into the system.")

