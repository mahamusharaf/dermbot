import chromadb
import os
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(SCRIPT_DIR, "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="dermatology")

def search(query, k=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    print(f"QUERY: {query}")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  [{meta['source']}] (distance: {dist:.3f})")
        print(f"  {doc[:150]}")
        print()

search("what are the risk factors for melanoma")
search("how is basal cell carcinoma treated")
search("what does a dermatofibroma look like")