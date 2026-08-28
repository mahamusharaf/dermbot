import chromadb, os
from sentence_transformers import SentenceTransformer, CrossEncoder

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db_bge"))
collection = client.get_collection(name="dermatology")

query = "acanthosis nigricans thickened dark patch neck insulin resistance diabetes"
query_embedding = embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=5)

print("=== raw vector search (pre-rerank) ===")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(meta["source"], "->", doc[:150])

pairs = [[query, doc] for doc in results["documents"][0]]
scores = reranker.predict(pairs)
reranked = sorted(zip(results["documents"][0], results["metadatas"][0], scores), key=lambda x: x[2], reverse=True)

print("\n=== after cross-encoder rerank ===")
for doc, meta, score in reranked:
    print(f"{score:.3f}  {meta['source']} -> {doc[:150]}")