import json

with open("chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

all_chunks = [c for c in all_chunks if c["source"] != "skincare_routine_order"]

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Removed bad chunks. chunks.json now has {len(all_chunks)} chunks")