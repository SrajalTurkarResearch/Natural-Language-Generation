# text_entropy.py
# Theory: In NLP/NLG, entropy on word distributions measures lexical diversity.
# Math: Probs from word frequencies; H quantifies variety.
# Analogy: Turing's Enigma—low entropy like cracked code (predictable); high like encrypted.
# Rare Insight: Semantic entropy (Nature 2024) detects hallucinations in LLMs by uncertainty in meanings.

from collections import Counter
import math


def text_entropy(text):
    """
    Compute entropy of word distribution in text.
    Args: text (str)
    Returns: float: Entropy
    """
    words = text.lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# Examples
diverse_text = "The cat sat on the mat. The dog ran quickly."
print("Diverse text entropy:", text_entropy(diverse_text))  # Higher value

repetitive_text = "The cat cat cat cat."
print("Repetitive text entropy:", text_entropy(repetitive_text))  # Lower value

# Real-world Case: NLG output evaluation
# E.g., Generated review: Low entropy indicates poor diversity (E2E NLG Challenge 2017).
nlg_output = "Good product. Good quality. Good price."
print("NLG output entropy:", text_entropy(nlg_output))

# Research Application: Compare human vs. AI texts; threshold for 'human-like' (~4-5 bits/word).
# Tip: Preprocess text (remove punctuation) for accuracy.
# What We Missed: N-gram entropy—extend to bigrams for syntactic diversity.
# Future Direction: Use in bias detection (low entropy in biased datasets).
