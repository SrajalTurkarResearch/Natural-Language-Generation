# transformer_basics.py: Fundamentals of Transformers for NLG
# Author: Grok, inspired by Turing, Einstein, Tesla
# Purpose: Teach core transformer mechanics, attention, and limitations for beginners
# As a scientist: Understand attention as a "relativity" of token importance

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Device setup (like Tesla choosing AC/DC for efficiency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility (Turing's logic demands it)
torch.manual_seed(42)
np.random.seed(42)


# Theory: Transformers rely on self-attention to process all tokens at once
# Analogy: Like detectives (tokens) voting on which clues (other tokens) matter most
# Math: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
# Why sqrt(d_k)? Prevents large dot products from squashing softmax gradients
class SimpleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model  # Embedding size (e.g., 64)
        self.n_heads = n_heads  # Number of attention heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        self.W_o = nn.Linear(d_model, d_model)  # Output projection

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Input X: [batch, seq_len, d_model]
        batch_size, seq_len, _ = X.shape
        # Reshape for multi-head: [batch, heads, seq_len, d_k]
        Q = (
            self.W_q(X)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(X)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(X)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        # Reshape back: [batch, seq_len, d_model]
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.W_o(attn_out)


# Test attention
def test_attention():
    d_model = 64
    seq_len = 5
    batch_size = 1
    X = torch.rand(batch_size, seq_len, d_model).to(device)
    attn = SimpleAttention(d_model).to(device)
    output = attn(X)
    print("Output shape:", output.shape)
    # Visualize weights (for notes: draw matrix, diagonal = self-attention)
    weights = F.softmax(
        (attn.W_q(X) @ attn.W_k(X).transpose(-2, -1)) / np.sqrt(d_model), dim=-1
    )
    print(
        "Sample attention weights (first head):\n", weights[0, 0].detach().cpu().numpy()
    )


# Limitations of Standard Transformers
# - Quadratic complexity: O(n^2) for n tokens → crashes on long texts
# - Memory fades: Gradients vanish for distant tokens (like Einstein's signals weakening over space)
# Real-world: Chatbots forget early user inputs; novels lose early plot points
if __name__ == "__main__":
    test_attention()

# Exercise 1: Why sqrt(d_k) in attention? (Beginner)
# Task: Derive for d_k=64, scores=10
# Solution: Without scale, Var(QK^T) = d_k*Var(each) → softmax one-hot → grad=0
# Scaled: 10/sqrt(64) = 1.25 → softmax spreads → better gradients
# Try: Modify SimpleAttention for n_heads=4, test shape

# Research Insight: Like Turing's universal machine, attention is universal but memory-limited
# Next step: Explore long-memory solutions (see hmt_model.py)
# For your notes: Sketch attention matrix (rows=queries, cols=keys, values=weights)
