import json

with open("chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

before = len(all_chunks)
all_chunks = [c for c in all_chunks if c["source"] != "diy_acne_masks"]
print(f"Removed {before - len(all_chunks)} chunks")

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)