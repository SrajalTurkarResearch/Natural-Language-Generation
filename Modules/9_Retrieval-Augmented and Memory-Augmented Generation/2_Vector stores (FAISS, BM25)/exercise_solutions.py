# exercise_solutions.py
# Purpose: Solve exercises on BM25 calculation and cosine/Euclidean visualization.
# Scientific Context: Like Turing verifying algorithms, these exercises solidify understanding.

import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

# Exercise 1: Manual BM25 Calculation
corpus_ex = ["apple", "apple banana"]
query_ex = "apple"
corpus_pre_ex = [re.findall(r"\w+", doc.lower()) for doc in corpus_ex]
query_t_ex = re.findall(r"\w+", query_ex.lower())
doc_lens_ex = [len(d) for d in corpus_pre_ex]
avgdl_ex = np.mean(doc_lens_ex)


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


score_ex = bm25_score(
    query_t_ex, corpus_pre_ex[1], corpus_pre_ex, doc_lens_ex[1], avgdl_ex
)
print("Exercise 1 - BM25 Score for Doc1:", score_ex)

# Exercise 2: Cosine vs Euclidean Visualization
v1 = np.array([1, 0])
v2 = np.array([0.9, 0.1])
cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
euc = np.linalg.norm(v1 - v2)
print("Exercise 2 - Cosine Similarity:", cos, "Euclidean Distance:", euc)

plt.figure(figsize=(6, 6))
plt.quiver(
    0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1, color="blue", label="v1"
)
plt.quiver(
    0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1, color="red", label="v2"
)
plt.xlim(-1, 1.5)
plt.ylim(-0.5, 1.5)
plt.title("Vector Comparison: Cosine vs Euclidean")
plt.grid(True)
plt.legend()
plt.show()

# Research Note: Verify calculations manually to build intuition, then scale with libraries.
