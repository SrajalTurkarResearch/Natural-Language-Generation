"""
rag_mini_project.py - Mini project to build a personal knowledge base RAG system.

Purpose: Create a RAG system for your personal notes, simulating a researcher's workflow
to query and summarize knowledge. This is a beginner-friendly project to practice RAG.

For Scientists: Treat this as a lab experiment—collect data, index it, and query like
Tesla testing a new circuit.
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from rag_setup import initialize_embeddings
from rag_knowledge_base import create_knowledge_base
from rag_retrieval_generation import run_rag_query


def build_personal_knowledge_rag(file_path):
    """
    Build a RAG system from a text file of personal notes.

    Args:
        file_path (str): Path to the notes file.

    Returns:
        FAISS: Indexed vector store.
    """
    # Load and split documents
    loader = TextLoader(file_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(docs)

    # Initialize embeddings and create vector store
    embeddings = initialize_embeddings()
    vectorstore = create_knowledge_base([doc.page_content for doc in texts], embeddings)
    return vectorstore


if __name__ == "__main__":
    # Example: Create a sample notes file (replace with your own)
    sample_notes = """
    Retrieval-Augmented Generation (RAG) is a method to improve LLMs.
    It uses external knowledge to reduce hallucinations.
    FAISS is a fast vector database for similarity search.
    """
    with open("notes.txt", "w") as f:
        f.write(sample_notes)

    # Build and query
    vectorstore = build_personal_knowledge_rag("notes.txt")
    query = "What is RAG?"
    response = run_rag_query(vectorstore, query, k=1)
