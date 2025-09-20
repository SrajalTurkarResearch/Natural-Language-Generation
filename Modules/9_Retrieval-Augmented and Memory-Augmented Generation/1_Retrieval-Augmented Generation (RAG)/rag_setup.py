"""
rag_setup.py - Initialize RAG system with embeddings for text processing.

Purpose: Set up the environment for Retrieval-Augmented Generation (RAG) by initializing
embedding models. This is the foundation for converting text to vectors, a critical step
in RAG's retrieval process.

For Aspiring Scientists: Understand embeddings as numerical representations of meaning,
like coordinates in a semantic space (Einstein's relativity for language).
"""

import os
from langchain.embeddings import HuggingFaceEmbeddings


def initialize_embeddings():
    """
    Initialize the embedding model for text-to-vector conversion.

    Returns:
        HuggingFaceEmbeddings: Model for embedding queries and documents.
    """
    # Set Hugging Face token if using API (replace with your token or remove for local)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"  # Optional

    # Use a lightweight, efficient model suitable for beginners
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embeddings ready. Model converts text to 384D vectors.")
    return embeddings


def test_embedding(embeddings, query="What is RAG?"):
    """
    Test embedding generation for a sample query.

    Args:
        embeddings: Initialized embedding model.
        query (str): Sample query to embed.

    Returns:
        list: Embedding vector for the query.
    """
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding shape: {len(query_embedding)}")
    return query_embedding


if __name__ == "__main__":
    # Initialize and test embeddings
    embeddings = initialize_embeddings()
    query_embedding = test_embedding(embeddings)
    print(f"Sample embedding (first 5 values): {query_embedding[:5]}")
