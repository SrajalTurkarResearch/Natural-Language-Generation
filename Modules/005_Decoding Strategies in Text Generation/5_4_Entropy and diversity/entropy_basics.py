# entropy_basics.py
# Theory: Shannon Entropy (H) measures uncertainty: H = -sum(p_i * log2(p_i))
# From Claude Shannon's 1948 paper—foundation of information theory.
# Analogy: Low H like a predictable experiment (Einstein's deterministic universe); high H like quantum uncertainty.
# Rare Insight: In NLG, low entropy signals repetitive outputs, akin to 'mode collapse' in GANs (research from Goodfellow, 2014).
# As Turing might compute: Use this to quantify diversity in generated text distributions.

import math


def shannon_entropy(probs):
    """
    Calculate Shannon Entropy.
    Args: probs (list of floats): Probabilities summing to 1.
    Returns: float: Entropy in bits.
    """
    return -sum(p * math.log2(p) for p in probs if p > 0)


# Example Calculations (from tutorial)
fair_coin = [0.5, 0.5]
print(
    "Fair coin entropy:", shannon_entropy(fair_coin)
)  # Expected: 1.0 (high uncertainty)

biased_coin = [0.99, 0.01]
print(
    "Biased coin entropy:", shannon_entropy(biased_coin)
)  # Expected: ~0.0808 (low uncertainty)

# NLG Application: Word probabilities in a sentence
# Real-world: Low entropy in chatbot responses indicates repetition (e.g., always "I'm sorry").
text_probs = [0.33, 0.17, 0.17, 0.17, 0.17]  # From "The cat sat. The dog ran."
print("Text entropy:", shannon_entropy(text_probs))  # Expected: ~2.25

# Research Direction: Extend to cross-entropy for model evaluation (e.g., perplexity = 2^H).
# Tip: Normalize probs if they don't sum to 1; handle zero probs to avoid log errors.
# Future: Integrate with quantum entropy for probabilistic NLG in emerging AI.
