"""
rag_retrieval_generation.py - Implement the RAG pipeline for query answering.

Purpose: Combine retrieval (FAISS) with generation (LLM) to answer queries accurately.
This is the heart of RAG, integrating external knowledge with language generation.

For Scientists: This pipeline mirrors scientific inquiry—retrieve evidence, synthesize answers
(Einstein's synthesis of observations into theory).
"""

from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from rag_setup import initialize_embeddings
from rag_knowledge_base import create_knowledge_base


def run_rag_query(vectorstore, query, k=2):
    """
    Run a RAG query: retrieve relevant documents and generate an answer.

    Args:
        vectorstore: FAISS vector store with indexed documents.
        query (str): User query to answer.
        k (int): Number of documents to retrieve.

    Returns:
        str: Generated response.
    """
    # Load LLM (using GPT-2 for simplicity; replace with stronger models like Llama)
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7})

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple context stuffing
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
    )

    # Execute query
    result = qa_chain.run(query)
    print(f"RAG Response: {result}")
    return result


if __name__ == "__main__":
    # Setup
    documents = [
        "Retrieval-Augmented Generation (RAG) combines retrieval and generation for accurate NLG.",
        "LLMs like GPT can hallucinate without external knowledge.",
        "Vector databases like FAISS enable fast similarity search.",
    ]
    embeddings = initialize_embeddings()
    vectorstore = create_knowledge_base(documents, embeddings)

    # Test query
    query = "What is RAG?"
    response = run_rag_query(vectorstore, query)
