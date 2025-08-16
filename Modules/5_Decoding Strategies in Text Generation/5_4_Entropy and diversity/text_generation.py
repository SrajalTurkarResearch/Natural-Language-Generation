# text_generation.py
# Theory: NLG needs diversity to avoid repetition; use probabilistic sampling.
# Analogy: Einstein's random walks in Brownian motion—generation as stochastic paths.
# Rare Insight: Diversity-promoting techniques (e.g., DP-GAN) force high entropy in training.

import numpy as np
from collections import Counter
import math  # For entropy calc

# Simple Markov Chain for generation
transitions = {
    "the": ["cat", "dog", "quick"],
    "cat": ["sat", "ran", "jumped"],
    "dog": ["barked", "ran"],
    "quick": ["fox"],
}


def generate_sentence(start="the", length=8):
    """
    Generate text using Markov chain.
    Args: start (str), length (int)
    Returns: str: Generated sentence
    """
    sentence = [start]
    for _ in range(length - 1):
        next_words = transitions.get(sentence[-1], ["."])
        sentence.append(np.random.choice(next_words))
    return " ".join(sentence)


# Generate and measure diversity
gen_text = generate_sentence()
print("Generated text:", gen_text)

# Compute entropy on generated words
words = gen_text.lower().split()
counts = Counter(words)
total = len(words)
probs = [count / total for count in counts.values() if count > 0]
entropy = -sum(p * math.log2(p) for p in probs if p > 0)
print("Generated entropy:", entropy)

# Application: Chatbots—high entropy for engaging responses.
# Real-world: Google Dialogflow uses similar for variety.
# Mini Project Idea: Vary transition probs to control diversity.
# Tip: Seed np.random for reproducibility in experiments.
# Next Steps: Integrate with RNNs (e.g., PyTorch) for advanced NLG.
