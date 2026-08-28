import trafilatura
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def scrape_article(url):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        favor_recall=True
    )
    return text

def clean_article(raw_text, content_start_phrase, end_markers):
    idx = raw_text.find(content_start_phrase)
    if idx == -1:
        raise ValueError(f"Start phrase not found: '{content_start_phrase}'")
    text = raw_text[idx:]
    for marker in end_markers:
        end_idx = text.find(marker)
        if end_idx != -1:
            text = text[:end_idx]
            break
    return text.strip()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n- ", "\n", ". ", " ", ""]
)

def chunk_document(text, source_name):
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "source": source_name, "chunk_id": i}
        for i, chunk in enumerate(chunks)
    ]

skincare_config = {
    "skin_care_tips": {
        "url": "https://www.aad.org/public/everyday-care/skin-care-secrets/routine/healthier-looking-skin",
        "start": "# 10 skin care secrets",
        "end_markers": ["#### Related AAD resources"],
    },
    "sunscreen_faqs": {
        "url": "https://www.aad.org/media/stats-sunscreen",
        "start": "# Sunscreen FAQs",
        "end_markers": ["Trullas C, Kohli I"],
    },
    "phytophotodermatitis": {
        "url": "https://dermnetnz.org/topics/phytophotodermatitis",
        "start": "Phytophotodermatitis, a form of plant dermatitis",
        "end_markers": ["ADVERTISEMENT"],
    },
    "topical_retinoids": {
        "url": "https://dermnetnz.org/topics/topical-retinoids",
        "start": "Topical retinoids are medications derived from vitamin A",
        "end_markers": ["ADVERTISEMENT"],
    },
}

# ---- scrape ----
raw_texts = {}
for name, cfg in skincare_config.items():
    print(f"Scraping {name}...")
    raw_texts[name] = scrape_article(cfg["url"])

# ---- clean + chunk ----
skincare_chunks = []
for name, cfg in skincare_config.items():
    clean_text = clean_article(raw_texts[name], cfg["start"], cfg["end_markers"])
    chunks = chunk_document(clean_text, name)
    skincare_chunks.extend(chunks)
    print(f"{name}: {len(chunks)} chunks")

print(f"\nTotal skincare chunks: {len(skincare_chunks)}")

# ---- merge into chunks.json ----
with open("chunks.json", "r", encoding="utf-8") as f:
    existing_chunks = json.load(f)

all_chunks = existing_chunks + skincare_chunks

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"chunks.json now has {len(all_chunks)} total chunks across {len(set(c['source'] for c in all_chunks))} classes")