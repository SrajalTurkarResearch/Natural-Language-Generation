# projects.py
# Theory: Projects bridge theory to practice—like Tesla's labs.
# Mini: Visualize text entropy.
# Major: Stub for RNN-based NLG with entropy eval.
# Rare Insight: In research, entropy balances coherence (use alongside BLEU scores).

import matplotlib.pyplot as plt
from collections import Counter
import math
import torch
import torch.nn as nn  # Requires PyTorch; install if needed.


def visualize_text_entropy(text):
    """
    Mini Project: Plot word freq and compute entropy.
    """
    words = text.lower().split()
    counts = Counter(words)
    probs = [count / len(words) for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    plt.bar(counts.keys(), counts.values())
    plt.title(f"Word Frequency - Entropy: {entropy:.2f} bits")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.show()


# Mini Project Example
visualize_text_entropy("Repeat repeat word diverse unique ideas.")


# Major Project Stub: Simple RNN for NLG
class SimpleNLGRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size)

    # Extend: Add forward, train on text, generate, compute entropy on outputs.


model = SimpleNLGRNN()
print("RNN Model Initialized:", model)

# Real-world: E-commerce descriptions—generate diverse ones, measure entropy.
# Case Study: Synthetic data gen for ML; high entropy improves model robustness.
# What We Missed: Conditional entropy—necessary for context-aware diversity (e.g., dependencies in equations).
# Tip: Log experiments; version control with Git for research reproducibility.
# Future: Quantum-inspired NLG with qutip lib for ultra-diverse outputs.
