# nlg_visualize.py: Visualization Tools for Embeddings
# Run: python nlg_visualize.py
# Requires: sentence-transformers, matplotlib, numpy, sklearn

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def visualize_embeddings(
    texts, labels, title="Embedding Space", model_name="all-MiniLM-L6-v2"
):
    """Visualize embeddings in 2D."""
    embed_model = SentenceTransformer(model_name)
    embeddings = [embed_model.encode(t) for t in texts]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, label in enumerate(labels):
        plt.text(reduced[i, 0], reduced[i, 1], label)
    plt.title(title)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()


if __name__ == "__main__":
    texts = [
        "The results are good.",
        "The outcomes are satisfactory.",
        "The results are okay.",
    ]
    labels = ["Input", "Formal", "Neutral"]
    visualize_embeddings(texts, labels)
