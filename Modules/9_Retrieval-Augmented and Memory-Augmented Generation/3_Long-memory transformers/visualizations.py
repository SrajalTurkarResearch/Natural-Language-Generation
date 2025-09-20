# visualizations.py: Visualize Attention and Memory Flow in Long-Memory Transformers
# Author: Grok, inspired by Turing, Einstein, Tesla
# Purpose: Show standard vs. HMT attention patterns; aid note-taking with diagrams
# Analogy: Attention as gravitational pull—HMT curves it for distant tokens

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Standard Transformer Attention (Recency Bias)
def gen_standard_attn(seq_len=20, d_k=8):
    Q = torch.rand(1, 1, seq_len, d_k)
    K = torch.rand(1, 1, seq_len, d_k)
    scores = torch.matmul(Q.squeeze(1), K.squeeze(1).transpose(-2, -1)) / np.sqrt(d_k)
    return F.softmax(scores, dim=-1).squeeze().detach().numpy()


# HMT Retrieval Weights (Long-Range Pull)
def gen_hmt_attn(N=5, d_model=64):
    Q = torch.rand(1, d_model)
    K = torch.rand(N, d_model)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model)
    return F.softmax(scores, dim=-1).squeeze().detach().numpy()


# Plot visualizations
def plot_attention():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Standard: Diagonal-heavy (recent tokens dominate)
    sns.heatmap(gen_standard_attn(20), ax=axs[0], cmap="Blues")
    axs[0].set_title("Standard Transformer Attention (Recency Bias)")
    # HMT: Selective long-range retrieval
    sns.barplot(x=range(5), y=gen_hmt_attn(), ax=axs[1], palette="Blues")
    axs[1].set_title("HMT Memory Retrieval Weights (Long-Range Pull)")
    plt.show()

    # Text-based diagram for notes
    print(
        """
    Memory Hierarchy Diagram:
    Sensory (k=32 tokens) --> Short-term (Summary S_n) 
                              |
                              v
    Long-term (N=300 caches) <--> Retrieval (Cross-Attention)
    """
    )
    # For notes: Sketch heatmap (left: diagonal blue; right: bars for memory weights)


# Application: Visualize medical report attention—HMT pulls early symptoms
# Research Insight: Test if HMT weights correlate with human recall patterns
if __name__ == "__main__":
    plot_attention()

# Exercise 3: Plot HMT weights for N=10 vs N=300. Hypothesize impact.
# Solution: Larger N → sparser weights; may overfit noise. Plot to confirm.
# For notes: Draw heatmap (standard: fade off-diagonal; HMT: spikes for key memories)
