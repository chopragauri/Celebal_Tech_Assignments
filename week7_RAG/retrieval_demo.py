"""
Standalone demo: shows retrieval results for sample queries on RLHF.pdf
Run: python retrieval_demo.py
"""

import os, re, warnings
warnings.filterwarnings("ignore")
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "RLHF.pdf")

def load_and_chunk(path, chunk_size=150, overlap=30):
    reader = PdfReader(path)
    raw = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    raw = re.sub(r"\s+", " ", raw).strip()
    words = raw.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    # Use RLHF.pdf from Desktop if not found locally
    doc = DOC_PATH if os.path.exists(DOC_PATH) else \
        "/Users/gaurichopra/Desktop/3RD YEAR/LLM/LLM.M3/RLHF.pdf"

    chunks = load_and_chunk(doc)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(chunks)

    print(f"Vector store: {mat.shape[0]} chunks x {mat.shape[1]:,} terms\n")

    queries = [
        "What is RLHF?",
        "reward modeling pairwise ranking",
        "PPO proximal policy optimization",
        "limitations of RLHF",
    ]

    for q in queries:
        q_vec = vec.transform([q])
        scores = cosine_similarity(q_vec, mat).flatten()
        top = scores.argsort()[::-1][:2]
        print(f"Query: '{q}'")
        for i in top:
            print(f"  [Chunk {i} | score={scores[i]:.4f}] {chunks[i][:120]}...")
        print()
