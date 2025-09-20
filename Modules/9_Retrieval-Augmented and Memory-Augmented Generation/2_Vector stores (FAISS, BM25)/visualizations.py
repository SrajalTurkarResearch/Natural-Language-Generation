# visualizations.py
# Purpose: Visualize vector spaces and BM25 scores for intuitive understanding.
# Scientific Context: Like Einstein's thought experiments, visuals clarify abstract vector concepts.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Demo corpus and BM25 scores (from bm25_implementation.py)
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick fox in the morning.",
    "Brown dogs are lazy in the sun.",
    "Jumps and runs in the field.",
]
# Example BM25 scores (replace with actual computation from bm25_implementation.py)
scores = [1.5, 1.2, 0.8, 0.3]  # Dummy for demo

# Vector Space Visualization (2D PCA-like)
# Fake 2D embeddings for simplicity (in practice, reduce high-dim embeddings)
vec2d = np.array([[0.1, 0.2], [0.15, 0.25], [0.8, 0.1], [0.9, 0.05]])
query2d = np.array([[0.12, 0.22]])

plt.figure(figsize=(8, 6))
plt.scatter(vec2d[:, 0], vec2d[:, 1], c="blue", label="Documents")
plt.scatter(query2d[0, 0], query2d[0, 1], c="red", marker="*", s=200, label="Query")
for i, txt in enumerate(["Doc0", "Doc1", "Doc2", "Doc3"]):
    plt.annotate(txt, (vec2d[i, 0], vec2d[i, 1]))
plt.title("Vector Space: Similarity as Proximity")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.show()

# BM25 Score Bar Plot
docs = ["Doc0", "Doc1", "Doc2", "Doc3"]
plt.figure(figsize=(8, 4))
plt.bar(docs, scores)
plt.title("BM25 Retrieval Scores")
plt.ylabel("Score")
plt.show()

# Research Note: Visualize real embeddings using PCA or t-SNE for high-dimensional data.
# Example: pca = PCA(n_components=2); reduced = pca.fit_transform(embeddings)
