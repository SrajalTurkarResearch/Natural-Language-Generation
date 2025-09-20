# llm_basics.py
# A foundational module for understanding LLMs and NLG
# Author: Inspired by Turing, Einstein, Tesla
# Date: September 20, 2025

"""
Theory: Introduction to LLMs and NLG
------------------------------------
Natural Language Generation (NLG) converts structured data into human-readable text.
Large Language Models (LLMs) are transformer-based neural networks trained on massive text datasets
to predict and generate language. Key equation: P(w_1, ..., w_n) = ∏ P(w_t | w_1:t-1).

Analogy (Einstein-inspired): LLMs are like a unified theory of language, synthesizing patterns
from vast data to generate coherent text, much like relativity unifies space-time.

Goal: Understand core mechanics and implement a basic text generation example.
"""

# Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Basic Transformer Attention Implementation
class SimpleAttention(nn.Module):
    """
    Implements single-head attention to demonstrate LLM's core mechanism.
    Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """

    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)


# Example: Generate Text with GPT-2
def generate_text(prompt, max_length=50):
    """
    Uses Hugging Face's GPT-2 to generate text.
    Logic: Tokenize prompt, predict next tokens, decode to text.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Visualization: Attention Weights (Simplified)
import matplotlib.pyplot as plt
import numpy as np


def plot_attention_weights():
    """
    Visualizes a mock attention weight matrix.
    Insight: Shows how LLMs focus on relevant tokens.
    """
    weights = np.random.rand(5, 5)  # Mock 5x5 attention matrix
    plt.imshow(weights, cmap="Blues")
    plt.title("Attention Weights Example")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.colorbar()
    plt.savefig("attention_weights.png")
    plt.close()


# Exercise: Predict Next Word Probabilities
def exercise_next_word_probs():
    """
    Exercise: Compute softmax probabilities for a mock vocabulary.
    Formula: P(w) = exp(z_w) / Σ exp(z_i)
    """
    logits = torch.tensor(
        [2.5, 1.8, 1.2]
    )  # Mock logits for ['blue', 'cloudy', 'clear']
    probs = F.softmax(logits, dim=0).numpy()
    words = ["blue", "cloudy", "clear"]
    print("Exercise 1: Next Word Probabilities")
    for w, p in zip(words, probs):
        print(f"{w}: {p:.3f}")


# Main Execution
if __name__ == "__main__":
    # Test Attention
    attn = SimpleAttention(d_model=64, d_k=64)
    Q = K = V = torch.rand(1, 5, 64)
    output = attn(Q, K, V)
    print(f"Attention Output Shape: {output.shape}")

    # Generate Text
    prompt = "The future of AI is"
    text = generate_text(prompt)
    print(f"Generated Text: {text}")

    # Visualize
    plot_attention_weights()

    # Exercise
    exercise_next_word_probs()

"""
Research Insight:
- LLMs' probabilistic nature allows generalization but risks hallucinations.
- As a scientist, explore fine-tuning GPT-2 on domain-specific data (e.g., arXiv papers).
"""
