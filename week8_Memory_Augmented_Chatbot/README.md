# Week 8 — Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG

> **Status: Still working on it 🚧**

## Project

Building a memory-augmented chatbot that combines:
- **Retrieval-Augmented Generation (RAG)** — FAISS vector search over scraped ML/AI articles
- **Knowledge Graph** — Neo4j storing entity relationships extracted via LLM
- **Long-Term Memory** — MongoDB storing user preferences and facts across sessions
- **LangGraph Orchestration** — Routes queries across RAG, graph, and memory nodes
- **Real-time Tools** — Wikipedia and arXiv API integrations
- **Evaluation** — RAGAS metrics (faithfulness, relevance, correctness)
- **FastAPI** — REST API with `/chat` and `/memory` endpoints

## Full Repo

[chopragauri/Memory-Augmented-Chatbot](https://github.com/chopragauri/Memory-Augmented-Chatbot)

## Stack
Python · LangGraph · Groq (LLaMA 3) · FAISS · Neo4j · MongoDB · FastAPI · RAGAS

## Progress

- [x] Project scaffold and architecture
- [x] Data pipeline (scraper → cleaner → embedder)
- [ ] Knowledge graph extraction and ingestion
- [ ] LangGraph agent integration
- [ ] Evaluation framework
- [ ] FastAPI deployment
