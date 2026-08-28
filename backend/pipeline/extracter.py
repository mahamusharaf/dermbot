import trafilatura

def scrape_dermnet_article(url):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        favor_recall=True
    )
    return text

def clean_dermnet_text(raw_text, content_start_phrase):
    idx = raw_text.find(content_start_phrase)
    if idx == -1:
        raise ValueError("Start phrase not found — check it matches exactly")
    text = raw_text[idx:]
    
    end_idx = text.find("ADVERTISEMENT")
    if end_idx != -1:
        text = text[:end_idx]
    
    return text.strip()
    
if __name__ == "__main__":
    melanoma_text = scrape_dermnet_article("https://dermnetnz.org/topics/melanoma")
    melanoma_clean = clean_dermnet_text(melanoma_text, "Melanoma, also referred to as")
    print(melanoma_clean[:300])
