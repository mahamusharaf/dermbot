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

# ---- re-scrape and re-clean sunscreen_faqs with the correct end marker ----
raw_text = scrape_article("https://www.aad.org/media/stats-sunscreen")
clean_text = clean_article(
    raw_text,
    "# Sunscreen FAQs",
    ["- Stern RS. Prevalence of a history of skin cancer"]
)
new_sunscreen_chunks = chunk_document(clean_text, "sunscreen_faqs")

print(f"New sunscreen_faqs chunk count: {len(new_sunscreen_chunks)}")
print("\n--- last chunk (sanity check — should be real content, not references) ---")
print(new_sunscreen_chunks[-1]["text"][:300])

# ---- replace old sunscreen_faqs chunks in chunks.json ----
with open("chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

all_chunks = [c for c in all_chunks if c["source"] != "sunscreen_faqs"]
all_chunks.extend(new_sunscreen_chunks)

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"\nchunks.json now has {len(all_chunks)} total chunks")