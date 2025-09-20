"""
rag_major_project.py - Major project for RAG in climate research.

Purpose: Build a RAG system to query climate-related documents, simulating a scientific
application for summarizing research findings.

For Researchers: This is your chance to apply RAG to a real-world domain, like Einstein
synthesizing climate data into actionable insights.
"""

from rag_setup import initialize_embeddings
from rag_knowledge_base import create_knowledge_base
from rag_retrieval_generation import run_rag_query


def climate_rag():
    """
    Build a RAG system for climate research with sample data.

    Returns:
        FAISS: Vector store for climate documents.
    """
    # Sample climate documents (replace with IPCC dataset or similar)
    climate_docs = [
        "Global warming has increased by 1.1°C since pre-industrial times.",
        "Renewable energy adoption mitigates CO2 emissions.",
    ]

    # Build knowledge base
    embeddings = initialize_embeddings()
    vectorstore = create_knowledge_base(climate_docs, embeddings)
    return vectorstore


if __name__ == "__main__":
    # Setup and query
    vectorstore = climate_rag()
    query = "Impacts of warming?"
    response = run_rag_query(vectorstore, query, k=2)
    print(f"Climate RAG Output: {response}")

    # Extend: Load real dataset (e.g., from HuggingFace datasets)
    # from datasets import load_dataset
    # dataset = load_dataset("climate_fever", split="train[:10]")
    # climate_docs = [item['claim'] for item in dataset]
