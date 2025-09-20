# Mini Project: Simple Text Generator
# Generates text using top-k sampling with a dummy model.
# Modify vocab and logits for experiments, e.g., scientific terms.

import numpy as np


def softmax(x):
    """Convert logits to probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def top_k_sampling(logits, k):
    """Top-k sampling function."""
    probs = softmax(logits)
    indices = np.argsort(probs)[::-1]
    top_k_indices = indices[:k]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= top_k_probs.sum()
    sampled_index = np.random.choice(top_k_indices, p=top_k_probs)
    return sampled_index


# Dummy model and vocab
vocab = {0: "The", 1: "cat", 2: "sat", 3: "on", 4: "mat"}


def dummy_logits(prev):
    """Dummy logits for demo."""
    return np.random.randn(5)


def generate_text(sampling_fn, param, length=10):
    """Generate text using given sampling function."""
    text = [0]  # Start with 'The'
    for _ in range(length):
        logits = dummy_logits(text[-1])
        next_token = sampling_fn(logits, param)
        text.append(next_token)
    return " ".join([vocab[t] for t in text])


# Run
if __name__ == "__main__":
    print("Generated text (top-k=3):", generate_text(top_k_sampling, 3))
