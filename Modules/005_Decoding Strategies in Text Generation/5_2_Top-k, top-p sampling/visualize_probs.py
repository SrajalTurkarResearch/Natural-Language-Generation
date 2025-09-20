# Visualize Probability Distributions
# Plots original and top-k renormalized probabilities.
# As a researcher, visualize to understand probability focus.

import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """Convert logits to probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Data
logits = np.array([3, 2, 1, 0, -1])
tokens = ["apple", "banana", "cat", "dog", "elephant"]
probs = softmax(logits)

# Plot original probs
plt.figure(figsize=(8, 4))
plt.bar(tokens, probs)
plt.title("Original Probability Distribution")
plt.xlabel("Tokens")
plt.ylabel("Probability")
plt.show()

# Plot top-k=3 renormalized probs
k = 3
indices = np.argsort(probs)[::-1][:k]
top_k_probs = probs[indices] / probs[indices].sum()
plt.figure(figsize=(8, 4))
plt.bar([tokens[i] for i in indices], top_k_probs)
plt.title("Top-k (k=3) Renormalized Probabilities")
plt.xlabel("Tokens")
plt.ylabel("Renormalized Probability")
plt.show()
