import chromadb
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- MiniLM setup ---
minilm_embedder = SentenceTransformer("all-MiniLM-L6-v2")
minilm_collection = minilm_client.get_collection(name="dermatology")

# --- BGE setup ---
bge_embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
minilm_client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "chroma_db"))
bge_client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "chroma_db_bge"))
bge_collection = bge_client.get_collection(name="dermatology")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def search_minilm(query, k=3, retrieve_n=10):
    query_embedding = minilm_embedder.encode([query]).tolist()
    results = minilm_collection.query(query_embeddings=query_embedding, n_results=retrieve_n)
    docs, metas = results["documents"][0], results["metadatas"][0]
    if not docs:
        return []
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    return [(doc, meta, float(score)) for doc, meta, score in reranked[:k]]

def search_bge(query, k=3, retrieve_n=10):
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = bge_embedder.encode([prefixed_query], normalize_embeddings=True).tolist()
    results = bge_collection.query(query_embeddings=query_embedding, n_results=retrieve_n)
    docs, metas = results["documents"][0], results["metadatas"][0]
    if not docs:
        return []
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    return [(doc, meta, float(score)) for doc, meta, score in reranked[:k]]

def compare(query):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")

    print("\n-- MiniLM --")
    for doc, meta, score in search_minilm(query):
        print(f"[{meta['source']}] score={score:.3f}")
        print(f"  {doc[:150]}")

    print("\n-- BGE --")
    for doc, meta, score in search_bge(query):
        print(f"[{meta['source']}] score={score:.3f}")
        print(f"  {doc[:150]}")

compare("what are the risk factors for melanoma")
compare("how is basal cell carcinoma treated")
compare("what does a dermatofibroma look like")
compare("what is eczema")