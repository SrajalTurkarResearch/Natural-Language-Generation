# Top-p (Nucleus) Sampling Implementation
# Implements top-p sampling, adaptive to probability distribution.
# Experiment with p to balance creativity vs. coherence.

import numpy as np


def softmax(x):
    """Convert logits to probabilities using softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def top_p_sampling(logits, p):
    """Perform top-p sampling: select nucleus where cumulative prob >= p."""
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_index = np.where(cumulative_probs >= p)[0][0] + 1
    nucleus_indices = sorted_indices[:cutoff_index]
    nucleus_probs = sorted_probs[:cutoff_index]
    nucleus_probs /= nucleus_probs.sum()
    sampled_index = np.random.choice(nucleus_indices, p=nucleus_probs)
    return sampled_index


# Example usage
if __name__ == "__main__":
    logits = np.array([3, 2, 1, 0, -1])
    tokens = ["apple", "banana", "cat", "dog", "elephant"]
    sampled_idx = top_p_sampling(logits, p=0.9)
    print(f"Sampled token: {tokens[sampled_idx]}")
