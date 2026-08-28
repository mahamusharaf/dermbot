import chromadb
import os
import json
from sentence_transformers import SentenceTransformer

with open("chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print(f"Loaded {len(all_chunks)} chunks")

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

db_path = os.path.join(os.getcwd(), "chroma_db_bge")  # new folder — don't mix with old MiniLM embeddings
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="dermatology")

texts = [c["text"] for c in all_chunks]
metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in all_chunks]
ids = [f"{c['source']}_{c['chunk_id']}" for c in all_chunks]

# BGE convention: passages/documents get embedded as-is, no prefix
embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

collection.upsert(
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas,
    ids=ids
)

print(f"Stored {collection.count()} chunks in the BGE vector store")