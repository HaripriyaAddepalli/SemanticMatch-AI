# SemanticMatch AI

**AI-Powered Semantic Candidate Ranking System**

Team Leader: Addepali Haripriya  
Team: SemanticMatch AI

## Overview
SemanticMatch AI uses advanced embeddings and hybrid search to rank candidates based on true semantic fit with a Job Description, going far beyond keyword matching.

## Features
- Semantic embeddings using Sentence Transformers
- Hybrid search (Semantic + BM25)
- Interactive Streamlit demo
- Explainable rankings
- Lightweight and fast

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Enter or modify the Job Description
2. View ranked candidates with similarity scores
3. Click on candidates to see detailed explanations

## Tech Stack
- Python
- Sentence-Transformers
- FAISS
- Streamlit
- scikit-learn (for BM25)

## Future Enhancements
- PDF resume upload & parsing
- LLM-powered natural language explanations
- More advanced reranking