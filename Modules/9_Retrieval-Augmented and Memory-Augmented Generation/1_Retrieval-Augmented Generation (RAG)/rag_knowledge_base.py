"""
rag_knowledge_base.py - Create and index a knowledge base for RAG.

Purpose: Build a vector store (knowledge base) using FAISS to store document embeddings.
This enables fast retrieval, a core component of RAG.

For Researchers: Think of the vector store as a library index (Turing's algorithmic efficiency),
where documents are stored as vectors for rapid similarity search.
"""

from langchain.vectorstores import FAISS
from rag_setup import initialize_embeddings


def create_knowledge_base(documents, embeddings):
    """
    Create a FAISS vector store from a list of documents.

    Args:
        documents (list): List of text strings to index.
        embeddings: HuggingFaceEmbeddings model for vectorization.

    Returns:
        FAISS: Indexed vector store.
    """
    # Index documents directly (for simplicity; extend with TextLoader for files)
    vectorstore = FAISS.from_texts(documents, embeddings)
    print(f"Knowledge base built. {len(documents)} documents indexed.")
    return vectorstore


if __name__ == "__main__":
    # Sample documents (replace with your own dataset)
    documents = [
        "Retrieval-Augmented Generation (RAG) combines retrieval and generation for accurate NLG.",
        "LLMs like GPT can hallucinate without external knowledge.",
        "Vector databases like FAISS enable fast similarity search.",
    ]

    # Initialize embeddings and create knowledge base
    embeddings = initialize_embeddings()
    vectorstore = create_knowledge_base(documents, embeddings)
