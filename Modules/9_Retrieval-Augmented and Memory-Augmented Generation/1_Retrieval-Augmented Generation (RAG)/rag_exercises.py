"""
rag_exercises.py - Practical exercises for learning RAG.

Purpose: Provide hands-on tasks to reinforce RAG concepts, with solutions to guide learning.
These exercises build your scientific intuition through experimentation.

For Scientists: Like Turing testing algorithms, experiment with parameters to uncover insights.
"""

from sklearn.metrics.pairwise import cosine_similarity
from rag_setup import initialize_embeddings
from rag_knowledge_base import create_knowledge_base
from rag_retrieval_generation import run_rag_query


def exercise_1_k1_impact():
    """
    Exercise 1: Modify k to 1 and observe impact on RAG response.
    """
    documents = [
        "Retrieval-Augmented Generation (RAG) combines retrieval and generation for accurate NLG.",
        "LLMs like GPT can hallucinate without external knowledge.",
        "Vector databases like FAISS enable fast similarity search.",
    ]
    embeddings = initialize_embeddings()
    vectorstore = create_knowledge_base(documents, embeddings)

    print("Running with k=1:")
    response = run_rag_query(vectorstore, "What is RAG?", k=1)
    print("Impact: More focused response, less context.")


def exercise_2_cosine_similarity():
    """
    Exercise 2: Compute cosine similarity between two embeddings.
    """
    embeddings = initialize_embeddings()
    emb1 = embeddings.embed_query("RAG is useful")
    emb2 = embeddings.embed_query("Retrieval helps LLMs")
    sim = cosine_similarity([emb1], [emb2])[0][0]
    print(f"Similarity between 'RAG is useful' and 'Retrieval helps LLMs': {sim}")


def exercise_3_reranking():
    """
    Exercise 3: Implement simple reranking (retrieve top-5, select top-2).
    """
    documents = [
        "Retrieval-Augmented Generation (RAG) combines retrieval and generation for accurate NLG.",
        "LLMs like GPT can hallucinate without external knowledge.",
        "Vector databases like FAISS enable fast similarity search.",
    ]
    embeddings = initialize_embeddings()
    vectorstore = create_knowledge_base(documents, embeddings)

    # Retrieve top-5 (here 3 for simplicity)
    retriever = vectorstore.as_retriever(search_kwargs={"k": len(documents)})
    docs = retriever.get_relevant_documents("What is RAG?")

    # Simple reranking: Select top-2 by embedding similarity
    query_emb = embeddings.embed_query("What is RAG?")
    scores = [
        cosine_similarity([query_emb], [embeddings.embed_query(doc.page_content)])[0][0]
        for doc in docs
    ]
    top_indices = np.argsort(scores)[-2:][::-1]
    top_docs = [docs[i].page_content for i in top_indices]

    print("Top-2 reranked documents:", top_docs)
    # Note: Integrate with LLM for generation in practice


if __name__ == "__main__":
    print("Exercise 1:")
    exercise_1_k1_impact()
    print("\nExercise 2:")
    exercise_2_cosine_similarity()
    print("\nExercise 3:")
    exercise_3_reranking()
