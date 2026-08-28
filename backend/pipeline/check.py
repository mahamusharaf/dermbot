import trafilatura
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

raw_text = scrape_article("https://www.aad.org/media/stats-sunscreen")

idx = raw_text.find("Stern RS. Prevalence")
print(raw_text[idx - 500 : idx + 100])