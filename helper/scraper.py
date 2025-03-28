import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
from helper.s3_utils import upload_to_s3  # ✅ Make sure this exists in your helper folder

def clean_filename_from_url(url):
    """
    Generate a clean, readable filename from a URL.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/").replace("/", "-")
    if not path:
        path = "index"
    return path + ".txt"

def scrape_text_from_url(url):
    """
    Scrape visible text content from a URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
        return text.strip()
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return None

def scrape_and_save_blog_text(url, s3_prefix):
    """
    Scrapes blog text from a URL and uploads it to S3.
    """
    text = scrape_text_from_url(url)
    if not text:
        return False

    filename = clean_filename_from_url(url)
    s3_key = os.path.join(s3_prefix, filename)

    return upload_to_s3(text, s3_key)
