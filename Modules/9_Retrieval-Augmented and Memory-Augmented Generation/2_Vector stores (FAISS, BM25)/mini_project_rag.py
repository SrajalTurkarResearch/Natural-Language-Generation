# mini_project_rag.py
# Purpose: Build a simple Retrieval-Augmented Generation (RAG) system with hybrid BM25 + cosine.
# Scientific Context: Like Turing's integration of logic and computation, this fuses sparse and dense retrieval.

import numpy as np
import torch
import torch.nn.functional as F
import re
from collections import Counter

# Demo corpus and query
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick fox in the morning.",
    "Brown dogs are lazy in the sun.",
    "Jumps and runs in the field.",
]
query = "quick fox brown"


# BM25 Functions (from bm25_implementation.py)
def preprocess(text):
    return re.findall(r"\w+", text.lower())


def compute_idf(corpus, term):
    N = len(corpus)
    n_term = sum(1 for doc in corpus if term in doc)
    if n_term == 0:
        return 0
    return np.log((N - n_term + 0.5) / (n_term + 0.5))


def bm25_score(
    query_tokens, doc_tokens, corpus_preprocessed, doc_len, avgdl, k1=1.2, b=0.75
):
    score = 0
    for qt in query_tokens:
        if qt in doc_tokens:
            f = doc_tokens.count(qt)
            idf = compute_idf(corpus_preprocessed, qt)
            numer = f * (k1 + 1)
            denom = f + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * (numer / denom)
    return score


# Compute BM25 scores
corpus_preprocessed = [preprocess(doc) for doc in corpus]
query_tokens = preprocess(query)
doc_lengths = [len(doc) for doc in corpus_preprocessed]
avgdl = np.mean(doc_lengths)
bm25_scores = []
for i, doc_tokens in enumerate(corpus_preprocessed):
    score = bm25_score(
        query_tokens, doc_tokens, corpus_preprocessed, doc_lengths[i], avgdl
    )
    bm25_scores.append(score)

# Cosine Similarity (from faiss_implementation.py, fallback)
np.random.seed(42)
d = 4
embeddings = np.random.rand(len(corpus), d).astype("float32")
query_emb = np.random.rand(1, d).astype("float32")
embeddings = F.normalize(torch.tensor(embeddings), dim=1).numpy()
query_emb = F.normalize(torch.tensor(query_emb), dim=1).numpy()
cos_scores = np.dot(embeddings, query_emb.T).flatten()

# Hybrid Retrieval
hybrid_scores = 0.5 * np.array(bm25_scores) + 0.5 * cos_scores
top_idx = np.argmax(hybrid_scores)

# Mock NLG Output
print(f"Hybrid Top Document: {corpus[top_idx]}")
print(
    "NLG Output: Retrieved context integrated into response: The quick brown fox is active!"
)

# Research Note: For real RAG, integrate with LLM (e.g., HuggingFace Transformers).
# Example: from transformers import pipeline; nlg = pipeline('text-generation')
