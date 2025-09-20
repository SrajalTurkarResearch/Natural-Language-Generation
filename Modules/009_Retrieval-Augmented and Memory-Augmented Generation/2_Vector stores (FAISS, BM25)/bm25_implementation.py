# bm25_implementation.py
# Purpose: Manual BM25 implementation for sparse vector retrieval in NLG.
# Scientific Context: Like Einstein deriving relativity, we compute relevance with mathematical precision.

import re
import numpy as np
from collections import Counter

# Demo corpus (from setup_and_data.py)
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick fox in the morning.",
    "Brown dogs are lazy in the sun.",
    "Jumps and runs in the field.",
]
query = "quick fox brown"


def preprocess(text):
    """Tokenize text into words, lowercase for consistency."""
    return re.findall(r"\w+", text.lower())


def compute_idf(corpus, term):
    """Calculate Inverse Document Frequency (IDF) for a term."""
    N = len(corpus)
    n_term = sum(1 for doc in corpus if term in doc)
    if n_term == 0:
        return 0  # Avoid division by zero
    return np.log((N - n_term + 0.5) / (n_term + 0.5))


def bm25_score(
    query_tokens, doc_tokens, corpus_preprocessed, doc_len, avgdl, k1=1.2, b=0.75
):
    """Compute BM25 score for a document given a query."""
    score = 0
    for qt in query_tokens:
        if qt in doc_tokens:
            f = doc_tokens.count(qt)  # Term frequency
            idf = compute_idf(corpus_preprocessed, qt)
            numer = f * (k1 + 1)
            denom = f + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * (numer / denom)
    return score


# Preprocess corpus
corpus_preprocessed = [preprocess(doc) for doc in corpus]
query_tokens = preprocess(query)
doc_lengths = [len(doc) for doc in corpus_preprocessed]
avgdl = np.mean(doc_lengths)

# Compute BM25 scores for all documents
scores = []
for i, doc_tokens in enumerate(corpus_preprocessed):
    score = bm25_score(
        query_tokens, doc_tokens, corpus_preprocessed, doc_lengths[i], avgdl
    )
    scores.append(score)

# Output results
print("BM25 Scores:", scores)
print("Ranked Documents (indices):", np.argsort(scores)[::-1])

# Scientific Note: High scores indicate relevance. For production, use rank_bm25 library.
# Example: from rank_bm25 import BM25Okapi; bm25 = BM25Okapi(corpus_preprocessed)
