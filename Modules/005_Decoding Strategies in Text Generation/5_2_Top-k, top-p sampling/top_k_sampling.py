# Top-k Sampling Implementation
# Implements the top-k sampling algorithm for NLG, as described in the tutorial.
# Run this to sample from a small vocabulary based on logits.
# As a scientist, modify k to observe effects on randomness.

import numpy as np


def softmax(x):
    """Convert logits to probabilities using softmax."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()


def top_k_sampling(logits, k):
    """Perform top-k sampling: select top k tokens, renormalize, sample."""
    probs = softmax(logits)
    indices = np.argsort(probs)[::-1]  # Sort descending
    top_k_indices = indices[:k]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= top_k_probs.sum()  # Renormalize
    sampled_index = np.random.choice(top_k_indices, p=top_k_probs)
    return sampled_index


# Example usage
if __name__ == "__main__":
    logits = np.array([3, 2, 1, 0, -1])  # Example logits
    tokens = ["apple", "banana", "cat", "dog", "elephant"]
    sampled_idx = top_k_sampling(logits, k=3)
    print(f"Sampled token: {tokens[sampled_idx]}")
