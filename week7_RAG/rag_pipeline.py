"""
Week 7 — RAG Document Question Answering System
Gauri Chopra | Celebal Technologies Data Science Internship

Pipeline:
  PDF → Extract Text → Word-based Chunks → TF-IDF Vectors → Cosine Similarity
                                                                     ↓
  User Query ─────────────────────────────────────────────→ Top-K Chunks → Groq LLM → Answer
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


# ── Step 1: Document Ingestion ────────────────────────────────────────────────

def load_pdf(path: str) -> tuple[str, int]:
    """Extract text from every page of a PDF. Returns (text, page_count)."""
    reader = PdfReader(path)
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return "\n".join(pages), len(reader.pages)


# ── Step 2: Text Chunking ─────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Args:
        chunk_size: words per chunk
        overlap:    words shared between consecutive chunks
    """
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


# ── Step 3: TF-IDF Vector Store ───────────────────────────────────────────────

def build_vector_store(chunks: list[str]):
    """Build a TF-IDF vector store over the chunks. Returns (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams for better term matching
        min_df=1,
        max_df=0.95,
    )
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


# ── Step 4: Retrieval ─────────────────────────────────────────────────────────

def retrieve(query: str, vectorizer, matrix, chunks: list[str], top_k: int = 3) -> list[dict]:
    """Embed query and retrieve top-K most similar chunks via cosine similarity."""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        {"chunk_id": int(i), "text": chunks[i], "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]


# ── Step 5: Generation ────────────────────────────────────────────────────────

def generate_answer(query: str, sources: list[dict], client: Groq) -> str:
    """Augment query with retrieved context and generate answer via Groq LLM."""
    context = "\n\n".join(
        f"[Source {i+1} | Chunk #{s['chunk_id']}]\n{s['text']}"
        for i, s in enumerate(sources)
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Q&A assistant. "
                    "Answer the user's question using ONLY the provided context. "
                    "Be concise and factual. "
                    "If the answer is not in the context, say: "
                    "'This information is not covered in the document.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        max_tokens=350,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── Full RAG Pipeline ─────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.chunks = []
        self.vectorizer = None
        self.matrix = None

    def ingest(self, pdf_path: str, chunk_size: int = 150, overlap: int = 30):
        raw_text, pages = load_pdf(pdf_path)
        print(f"Loaded {pages} pages | {len(raw_text):,} chars | {len(raw_text.split()):,} words")
        self.chunks = chunk_text(raw_text, chunk_size, overlap)
        self.vectorizer, self.matrix = build_vector_store(self.chunks)
        print(f"Built vector store: {len(self.chunks)} chunks × {self.matrix.shape[1]:,} terms")

    def ask(self, query: str, top_k: int = 3) -> dict:
        sources = retrieve(query, self.vectorizer, self.matrix, self.chunks, top_k)
        if not sources:
            return {"question": query, "answer": "No relevant content found.", "sources": []}
        answer = generate_answer(query, sources, self.client)
        return {"question": query, "answer": answer, "sources": sources}

    def ask_and_display(self, query: str):
        result = self.ask(query)
        print(f"\nQ: {result['question']}")
        print("=" * 65)
        print(f"A: {result['answer']}")
        if result["sources"]:
            print(f"\nRetrieved {len(result['sources'])} source(s):")
            for s in result["sources"]:
                print(f"  [Chunk #{s['chunk_id']} | Score: {s['score']:.3f}] {s['text'][:90]}...")
        print("=" * 65)


if __name__ == "__main__":
    pipeline = RAGPipeline(groq_api_key=os.environ["GROQ_API_KEY"])
    pipeline.ingest("RLHF.pdf")

    questions = [
        "What is RLHF?",
        "How does reward modeling work in RLHF?",
        "What is PPO and why is it used?",
        "What are the main steps in the RLHF training pipeline?",
        "What are the limitations of RLHF?",
    ]
    for q in questions:
        pipeline.ask_and_display(q)
