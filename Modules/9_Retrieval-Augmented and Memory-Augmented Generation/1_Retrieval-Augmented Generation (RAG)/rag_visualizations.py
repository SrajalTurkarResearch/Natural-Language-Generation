"""
rag_visualizations.py - Visualize RAG embeddings and pipeline.

Purpose: Create plots to understand embeddings in 2D space using PCA, aiding intuition
for semantic similarity in RAG's retrieval process.

For Researchers: Visuals are like Einstein's thought experiments—simplify complex vector
spaces into intuitive 2D representations for analysis.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from rag_setup import initialize_embeddings
from rag_knowledge_base import create_knowledge_base


def plot_embeddings(documents, query, embeddings):
    """
    Plot query and document embeddings in 2D using PCA.

    Args:
        documents (list): List of documents to embed.
        query (str): Query to embed.
        embeddings: HuggingFaceEmbeddings model.
    """
    # Embed query and documents
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = embeddings.embed_documents(documents)
    all_embeddings = np.array([query_embedding] + doc_embeddings)

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[0, 0], reduced[0, 1], color="red", label="Query", s=100)
    for i, (x, y) in enumerate(reduced[1:]):
        plt.scatter(x, y, color="blue", label=f"Doc {i+1}" if i == 0 else None)
        plt.annotate(f"Doc{i+1}", (x, y))
    plt.title("RAG Embeddings in 2D Space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Closer points = higher similarity. RAG retrieves nearest neighbors.")


if __name__ == "__main__":
    # Sample data
    documents = [
        "Retrieval-Augmented Generation (RAG) combines retrieval and generation for accurate NLG.",
        "LLMs like GPT can hallucinate without external knowledge.",
        "Vector databases like FAISS enable fast similarity search.",
    ]
    query = "What is RAG?"

    # Initialize and plot
    embeddings = initialize_embeddings()
    vectorstore = create_knowledge_base(documents, embeddings)
    plot_embeddings(documents, query, embeddings)
