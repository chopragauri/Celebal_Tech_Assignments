# Week 7 — Document Question Answering System (RAG)

**Gauri Chopra | Celebal Technologies Data Science Internship**

A Retrieval-Augmented Generation (RAG) system that answers questions from custom PDF documents — built entirely with free tools, no paid API required.

---

## Pipeline

```
PDF ──► Extract Text ──► Word Chunks ──► TF-IDF Vectors ──► Vector Store
                                                                   │
User Query ─────────────────────────────────────────────► Cosine Similarity
                                                                   │
                                                             Top-K Chunks
                                                                   │
                                                         Groq LLM (Llama 3.1 8B)
                                                                   │
                                                                Answer
```

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF loading | `pypdf` |
| Text chunking | Custom word-based with overlap |
| Embeddings | TF-IDF (unigrams + bigrams) via `sklearn` |
| Vector store | Sparse TF-IDF matrix |
| Similarity search | Cosine similarity via `sklearn` |
| LLM generation | **Groq API — Llama 3.1 8B (free tier)** |

## Document Used

**RLHF.pdf** — personal LLM module notes on Reinforcement Learning from Human Feedback (32 pages)

## Sample Q&A Results

| Question | Answer |
|----------|--------|
| What is RLHF? | Uses human preferences as the ultimate reward signal to fine-tune LLMs |
| How does reward modeling work? | Trains on pairwise human rankings → outputs scalar reward score |
| What is PPO? | Stable RL algorithm that prevents catastrophic forgetting during fine-tuning |
| Main steps in RLHF? | SFT → Reward Model Training → RL Fine-Tuning (PPO) |
| Limitations of RLHF? | Needs thousands of human labelers; reward hacking risk |

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # free at console.groq.com
python rag_pipeline.py
```

Or open `week7_gauriChopra.ipynb` in Jupyter.

## Key Learnings

- How RAG combines retrieval and generation to reduce hallucinations
- TF-IDF vector representations and cosine similarity search
- Word-based chunking with overlap prevents answer boundary issues
- Prompt engineering to keep LLM answers grounded in retrieved context
