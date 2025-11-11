# math_utils.py
"""
Mathematical Utilities for Empathy Research
Softmax, cross-entropy, attention visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores to probabilities."""
    exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
    return exp_scores / np.sum(exp_scores)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute loss for training emotion classifiers."""
    return -np.sum(y_true * np.log(y_pred + 1e-8))


def plot_emotion_probs(
    emotions: list, probs: list, title: str = "Emotion Probabilities"
):
    """Visualize emotion distribution."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        emotions,
        probs,
        color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0", "#ff9f99"],
    )
    plt.title(title)
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    for bar, p in zip(bars, probs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{p:.3f}",
            ha="center",
            va="bottom",
        )
    plt.show()


# === DEMO ===
if __name__ == "__main__":
    scores = np.array([2.0, 1.0, 0.1, -0.5])
    probs = softmax(scores)
    plot_emotion_probs(["joy", "sadness", "anger", "fear"], probs)
