import chromadb
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(SCRIPT_DIR, "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="dermatology")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def search(query, k=3, retrieve_n=10, use_reranker=True):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=retrieve_n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not use_reranker:
        return list(zip(docs, metas))[:k]

    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    return [(doc, meta) for doc, meta, score in reranked[:k]]

def generate_answer(query):
    retrieved = search(query)

    context = "\n\n".join(
        f"[Source: {meta['source']}]\n{doc}" for doc, meta in retrieved
    )

    prompt = f"""You are a dermatology information assistant. Answer the question using ONLY the context provided below. If the context doesn't contain enough information to answer, say so clearly — do not use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

answer = generate_answer("what are the risk factors for melanoma")
print(answer)