# code_guides.py: Practical Code for Grounded NLG
# This file has working Python code to show grounded NLG.
# Run in Python 3.8+ with numpy and transformers installed.
# Each section links to theory.py for understanding.

# Import libraries
import numpy as np

try:
    from transformers import pipeline
except ImportError:
    print("Install transformers: pip install transformers torch")


# 2.1 Simple Template-Based Grounded NLG
# Theory: Fills a sentence with a fact to ensure truth (see theory.py #3).
def template_nlg(fact_dict):
    fact = fact_dict.get("fact", "unknown")
    return f"The fact is: {fact}"


# Test
fact_dict = {"fact": "The capital of France is Paris."}
print(
    "Template NLG:", template_nlg(fact_dict)
)  # Output: The fact is: The capital of France is Paris.

# 2.2 Simulated RAG with Cosine Similarity
# Theory: Finds best fact match and uses it (see theory.py #4).
# Simulate fact database
facts = [
    {"text": "France capital is Paris", "embedding": np.array([0.9, 0.5])},
    {"text": "UK capital is London", "embedding": np.array([0.2, 0.9])},
]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def rag_nlg(question, facts):
    # Simulate question embedding (in real systems, use BERT)
    q_embed = np.array([0.8, 0.6])  # For 'capital France'
    best_score = -1
    best_fact = None
    for fact in facts:
        score = cosine_similarity(q_embed, fact["embedding"])
        if score > best_score:
            best_score = score
            best_fact = fact["text"]
    return f"Based on facts: {best_fact}"


# Test
print(
    "RAG NLG:", rag_nlg("What is France capital?", facts)
)  # Output: Based on facts: France capital is Paris


# 2.3 Advanced: Pre-Trained Model
# Theory: Uses a real AI model with grounding (see theory.py #5).
# Note: Requires transformers and torch.
def grounded_generate(question, fact):
    try:
        generator = pipeline("text-generation", model="distilgpt2")
        prompt = f"Question: {question}\nFact: {fact}\nAnswer based only on the fact:"
        result = generator(prompt, max_length=50, num_return_sequences=1)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error: Install transformers or check GPU. {e}"


# Test
fact = "The capital of France is Paris."
print(
    "Advanced NLG:", grounded_generate("What is the capital of France?", fact)
)  # Output: Varies, includes Paris

# Reflection: Run these. How does grounding change the output? Try new facts!
