from docx import Document

def extract_text_from_docx(file):
    """
    Extracts plain text from a .docx file-like object (uploaded via Streamlit).
    
    :param file: Uploaded file-like object (e.g., from st.file_uploader)
    :return: Extracted text as a string
    """
    try:
        doc = Document(file)
        text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        print(f"❌ Failed to extract text from .docx: {e}")
        return None
