# math_and_optimization.py
# Module for mathematical foundations of LLMs
# Date: September 20, 2025

"""
Theory: Mathematical Foundations
-------------------------------
LLMs rely on probability (softmax for next-word prediction) and optimization
(cross-entropy loss, Adam optimizer).

Key Equations:
- Softmax: P(w) = exp(z_w) / Σ exp(z_i)
- Cross-Entropy Loss: L = -Σ y log(p)
- Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V

Analogy (Turing-inspired): Like a universal machine computing probabilities,
LLMs optimize language generation via mathematical rigor.

Goal: Implement and derive key equations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# Softmax Implementation
def compute_softmax(logits):
    """
    Computes softmax probabilities for given logits.
    Derivation: P(w) = exp(z_w) / Σ exp(z_i)
    """
    exp_logits = torch.exp(logits)
    return exp_logits / torch.sum(exp_logits)


# Loss Function
def compute_loss(predicted, target):
    """
    Computes cross-entropy loss.
    Formula: L = -Σ y log(p)
    """
    return -torch.sum(target * torch.log(predicted))


# Example Calculation
def example_calculation():
    """
    Demonstrates softmax and loss for a mock vocabulary.
    Prompt: "The sky is" -> Predict ['blue', 'cloudy', 'clear']
    """
    logits = torch.tensor([2.5, 1.8, 1.2])  # Mock logits
    target = torch.tensor([1.0, 0.0, 0.0])  # True word: 'blue'
    probs = compute_softmax(logits)
    loss = compute_loss(probs, target)
    print(f"Probabilities: {probs.numpy()}")
    print(f"Loss: {loss.item():.3f}")


# Visualization: Loss Curve
def plot_loss_curve():
    """
    Simulates training loss over epochs.
    Insight: Shows optimization convergence.
    """
    epochs = range(1, 11)
    losses = [5 / (e + 1) for e in epochs]  # Mock loss decrease
    plt.plot(epochs, losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.close()


# Exercise: Derive Softmax Gradient
def exercise_softmax_gradient():
    """
    Exercise: Compute gradient of softmax for a single logit.
    Formula: ∂P_i/∂z_j = P_i (δ_ij - P_j)
    """
    logits = torch.tensor([2.5, 1.8, 1.2], requires_grad=True)
    probs = compute_softmax(logits)
    probs[0].backward()  # Backprop on P(blue)
    print("Exercise 3: Softmax Gradient")
    print(f"Gradient of logits: {logits.grad.numpy()}")


# Main Execution
if __name__ == "__main__":
    example_calculation()
    plot_loss_curve()
    exercise_softmax_gradient()

"""
Research Insight:
- Study Adam optimizer equations for LLM training.
- Explore reinforcement learning (RLHF) for tool-use optimization.
"""
