import requests
from bs4 import BeautifulSoup
from docx import Document
import os
import re
import hashlib
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# def clean_filename(url, max_length=100):
#     """Generate a clean, readable filename from a URL."""
#     filename = url.rstrip("/").split("/")[-1]  # Extract last part of the URL
#     filename = re.sub(r'[^a-zA-Z0-9-_]', '-', filename)  # Replace special characters with "-"
#     filename = re.sub(r'-+', '-', filename).strip("-")  # Remove duplicate "--"
    
#     # Ensure filenames are not cut off at bad places
#     if len(filename) > max_length:
#         filename = filename[:max_length].rsplit("-", 1)[0]  # Trim at last '-' for readability
    
#     return filename.lower()



# Client's own blogs for retrieval
OUR_BLOGS = [
    # BnB Website Blogs
    "https://www.bnbcontentstudio.com/internal-communications-video-metrics/",
    "https://www.bnbcontentstudio.com/pitfalls-to-avoid-b2b-video-communications/",
    "https://www.bnbcontentstudio.com/b2b-tiktok/",
    "https://www.bnbcontentstudio.com/b2b-videos-corporate-governance/",
    "https://www.bnbcontentstudio.com/video-business-communications/",
    "https://www.bnbcontentstudio.com/b2b-humanize/",
    "https://www.bnbcontentstudio.com/the-ultimate-guide-to-b2b-video-distribution-where-and-how-to-share/",
    "https://www.bnbcontentstudio.com/2024-b2b-video-trends/",
    "https://www.bnbcontentstudio.com/b2b-video-metrics-that-matter/",

    # BnB LinkedIn Blogs
    "https://www.linkedin.com/pulse/how-ai-transforming-video-production-bnbcontentstudio-3jate/",
    "https://www.linkedin.com/pulse/video-production-experts-predict-2025s-biggest-trends-wanee/",
    "https://www.linkedin.com/pulse/b2b-video-production-made-simple-tips-tricks-success-iq2me/",
    "https://www.linkedin.com/pulse/how-best-showcase-your-companys-sustainability-initiatives-1mvge/",
    "https://www.linkedin.com/pulse/top-5-video-marketing-trends-every-b2c-marketer-should-7fm1e/",
    "https://www.linkedin.com/pulse/role-video-b2b-marketing-strategies-bnbcontentstudio-hdg1e/",
    "https://www.linkedin.com/pulse/how-video-can-help-enhance-your-brands-identity-bnbcontentstudio-cp3ge/",
    "https://www.linkedin.com/pulse/power-customer-testimonial-videos-bnbcontentstudio-apx2e/",
    "https://www.linkedin.com/pulse/why-companies-need-video-marketing-now-more-than-ever-muqle/",
    "https://www.linkedin.com/pulse/engage-convert-succeed-power-b2b-video-lead-generation-ep1pe/",
    "https://www.linkedin.com/pulse/5-essential-questions-aspect-ratio-bnbcontentstudio/",
]

# Blogs for style inspiration
STYLE_BLOGS = [
    # FARM Marketing Agency
    "https://www.growwithfarm.com/our-take/how-this-weekly-exercise-can-help-you-sell-more-customers/",
    "https://www.growwithfarm.com/our-take/what-never-to-say-at-the-start-of-your-sales-pitch/",
    "https://www.growwithfarm.com/our-take/how-to-instantly-multiply-your-number-of-customer-testimonials/",
    "https://www.growwithfarm.com/our-take/two-words-that-give-your-marketing-extraordinary-power/",

    # The Marketing Millennials
    "https://themarketingmillennials.com/articles/2025-01-30/%f0%9f%94%a5-how-ai-is-changing-seo/",
    "https://themarketingmillennials.com/articles/2025-01-28/%f0%9f%94%a5-how-to-leverage-scarcity/",
    "https://themarketingmillennials.com/articles/2024-12-24/how-coca-cola-makes-holiday-magic/",

    # Other Agencies and Marketing Blogs
    "https://wearefevr.com/the-influence-of-swiss-design/",
    "https://wearefevr.com/top-10-motion-graphics-styles/",
    "https://www.umault.com/insights/why-storytelling-is-the-secret-ingredient-in-b2b-marketing",
    "https://contentmarketinginstitute.com/articles/b2b-content-marketing-trends-research/",
    "https://www.marketingprofs.com/articles/2021/45481/five-b2b-video-marketing-tips-to-boost-sales-in-2021",
    "https://www.marketingprofs.com/articles/2023/49436/b2b-product-video-best-practices",
    "https://www.boathouseinc.com/insights//healthcare-marketing-key-trends-shaping-2025",
    "https://www.boathouseinc.com/insights/aligning-communications-social-excellence-an-asymmetric-world",
    "https://www.boathouseinc.com/insights/business-of-marketing",
    "https://www.profgalloway.com/killing-the-cat/",
    "https://www.profgalloway.com/olympic-moments/",
    "https://www.profgalloway.com/addiction-economy/",
    "https://www.linkedin.com/pulse/creative-production-model-how-manage-projects-less-time-lower/?trackingId=46KnBU6DdyUlyFddUxxPKg%3D%3D",
    "https://www.linkedin.com/pulse/building-future-wellcom-worldwide-vision-integrated-creative/?trackingId=RcCHOfWuPa5YczCVt4B7ng%3D%3D",
    ## Additional blogs
    "https://www.morningbrew.com/sponsored/stackadapt/marketers-meet-your-match?gad_source=1&gclid=CjwKCAjwnPS-BhBxEiwAZjMF0mX7ZlBmWOK3wdkPxeWGDFZtTyoeviEk-IRL2BMzBIyu8GAI_JbVzBoCaB8QAvD_BwE",
    "https://www.thecurrent.com/culture-3-takes-how-ai-is-overhauling-advertising-marketecture-live",
    "https://www.thedrum.com/open-mic/how-tacky-tiktoks-and-kitschy-campaigns-deliver-authenticity",
    "https://www.linkedin.com/business/marketing/blog/content-marketing/the-ultimate-guide-to-improve-your-b2b-content-marketing-strategy",
    "https://www.blendb2b.com/blog/the-7-best-b2b-marketing-strategies",
    "https://hingemarketing.com/blog/story/10-essential-b2b-marketing-strategies-to-grow-your-professional-services-fi",
    "https://www.blendb2b.com/blog/does-seo-still-work-in-2025",
    "https://www.toptal.com/external-blogs/growth-collective/killer-b2b-marketing-strategies",
    "https://growthmodels.co/funny-marketing-campaigns/"
    "https://www.lyfemarketing.com/blog/graphic-design-tips/",
    "https://www.linkedin.com/posts/annhandley_is-messaging-the-new-email-this-stopped-activity-7304576837198774273-czME?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAXXTrgBlWw6gN40vfduyXo6NRB0rN2TOzA",
    "https://www.linkedin.com/posts/hnshah_ai-is-not-the-future-of-work-its-the-return-activity-7309982364179304448-U2mv?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAXXTrgBlWw6gN40vfduyXo6NRB0rN2TOzA",
    "https://www.linkedin.com/posts/rajakarmani_branding-vs-marketing-activity-7308390198856286210-RZjS?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAXXTrgBlWw6gN40vfduyXo6NRB0rN2TOzA",
    "https://www.linkedin.com/posts/leedensmer_5-content-marketing-hills-i-will-die-on-activity-7308089035455893504-FnRo?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAXXTrgBlWw6gN40vfduyXo6NRB0rN2TOzA",
    "https://www.linkedin.com/posts/armonshokravi_this-ikea-severance-ad-genius-why-do-activity-7309914196773675008-4Lyn?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAXXTrgBlWw6gN40vfduyXo6NRB0rN2TOzA",
    "https://www.linkedin.com/pulse/marketing-different-generations-annie-mai-hodge/?trackingId=UvHgvDh3acSoCEWbL3BQUg%3D%3D",
    "https://www.linkedin.com/pulse/why-your-business-needs-social-media-marketing-seriously-hodge/?trackingId=iQ61hGXiQPS04FPnsEl%2Fwg%3D%3D",
    "https://www.linkedin.com/posts/jamesmulvey_creatives-live-here-david-ogilvy-said-activity-7309991299669995520-5-eH?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAXXTrgBlWw6gN40vfduyXo6NRB0rN2TOzA",
    "https://www.thecurrent.com/culture-3-takes-how-ai-is-overhauling-advertising-marketecture-live",
    "https://www.thedrum.com/open-mic/how-tacky-tiktoks-and-kitschy-campaigns-deliver-authenticity",
    "https://www.blendb2b.com/blog/the-7-best-b2b-marketing-strategies",
    "https://hingemarketing.com/blog/story/10-essential-b2b-marketing-strategies-to-grow-your-professional-services-fi",
    "https://www.blendb2b.com/blog/does-seo-still-work-in-2025",
    "https://www.toptal.com/external-blogs/growth-collective/killer-b2b-marketing-strategies",
    "https://growthmodels.co/funny-marketing-campaigns/",
    "https://www.lyfemarketing.com/blog/graphic-design-tips/",





]


# Helper Function: Clean Filenames
def clean_filename(url, max_length=100):
    filename = url.rstrip("/").split("/")[-1]  
    filename = re.sub(r'[^a-zA-Z0-9-_]', '-', filename)  
    filename = re.sub(r'-+', '-', filename).strip("-")  

    if len(filename) > max_length:
        filename = filename[:max_length].rsplit("-", 1)[0]  

    return filename.lower()

# Helper Function: Scrape Blog Text
def extract_text_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        print(f"🔄 Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])

        if text.strip():
            print(f"✅ Scraped: {url}")
        else:
            print(f"⚠️ No content found in {url}")

        return text.strip() if text else None
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

# Helper Function: Upload File to S3
def upload_to_s3(content, s3_path):
    try:
        s3_client.put_object(Body=content, Bucket=S3_BUCKET, Key=s3_path)
        print(f"📂 Uploaded to S3: {s3_path}")
    except Exception as e:
        print(f"❌ S3 Upload Error: {e}")

# Process Blogs & Upload to S3
def save_blog_text(url_list, category):
    for url in url_list:
        text = extract_text_from_url(url)
        if text:
            filename = clean_filename(url) + ".txt"
            s3_path = f"blogs/{category}/{filename}"
            upload_to_s3(text, s3_path)

# Process Brand Documents
def extract_text_from_doc(doc_path):
    try:
        print(f"🔍 Extracting text from {doc_path}...")
        doc = Document(doc_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"❌ Error reading {doc_path}: {e}")
        return None

def save_brand_documents():
    brand_docs = {
        "brand_voice": "data/BNB Brand Voice Document 030425.docx",
        "elevator_pitch": "data/BNB Elevator Pitch Document 030425.docx"
    }
    for key, path in brand_docs.items():
        if os.path.exists(path):
            text = extract_text_from_doc(path)
            if text:
                s3_path = f"docs/{key}.txt"
                upload_to_s3(text, s3_path)
                print(f"✅ Uploaded: {key}.txt")
        else:
            print(f"⚠️ File Not Found: {path}")

# MAIN EXECUTION
if __name__ == "__main__":
    print("🔄 Scraping & Uploading BnB Blogs...\n")
    save_blog_text(OUR_BLOGS, "our")

    print("🔄 Scraping & Uploading Style Blogs...\n")
    save_blog_text(STYLE_BLOGS, "style")

    print("🔄 Uploading Brand Voice Documents...\n")
    save_brand_documents()

    print("✅ Data ingestion complete!")