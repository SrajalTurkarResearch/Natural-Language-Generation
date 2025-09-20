# major_project.py: Grounded Q&A System
# This builds a Q&A system with fact grounding.
# Theory: Uses fact-based grounding with RAG-like approach (see theory.py #4).
# Run with numpy: pip install numpy

import numpy as np

# Fact database
fact_db = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "embedding": np.array([0.8, 0.6]),
    },
    {
        "question": "What is the capital of UK?",
        "answer": "London",
        "embedding": np.array([0.2, 0.9]),
    },
]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def qa_nlg(query):
    # Simulate query embedding
    q_embed = np.array([0.8, 0.6])  # For 'capital France'
    best_score = -1
    best_answer = "Sorry, I don’t know."
    for fact in fact_db:
        score = cosine_similarity(q_embed, fact["embedding"])
        if score > best_score:
            best_score = score
            best_answer = fact["answer"]
    return f"Answer: {best_answer}"


def qa_nlg2(query):
    # Simulate query embedding
    q_embed = np.array([0.2, 0.9])  # For 'capital France'
    best_score = -1
    best_answer = "Sorry, I don’t know."
    for fact in fact_db:
        score = cosine_similarity(q_embed, fact["embedding"])
        if score > best_score:
            best_score = score
            best_answer = fact["answer"]
    return f"Answer: {best_answer}"


# Test
print("Major Project:", qa_nlg("Capital of France?"))  # Output: Answer: Paris
print("MP :", qa_nlg2("UK capital"))

# Task for You:
# 1. Add 10 more facts to fact_db (e.g., science facts).
# 2. Test with new questions.
# 3. Research Idea: Add a truth-checker (compare answer to fact text).
# 4. Think: How could this scale for real science data (e.g., biology)?
