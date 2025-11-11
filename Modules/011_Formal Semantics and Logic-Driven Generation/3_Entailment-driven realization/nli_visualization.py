# nli_visualization.py
# Purpose: Visualize NLI softmax probabilities as a bar chart
# For aspiring scientists: Understand how NLI models assign probabilities
# Dependencies: matplotlib, numpy (pip install matplotlib numpy)

import matplotlib.pyplot as plt
import numpy as np


def plot_nli_probabilities(logits):
    """
    Plot bar chart of NLI softmax probabilities.
    Args:
        logits (list): Scores for entailment, neutral, contradiction
    """
    # Calculate softmax
    exp_logits = [np.exp(x) for x in logits]
    sum_exp = sum(exp_logits)
    probs = [x / sum_exp for x in exp_logits]

    # Plot
    plt.bar(["Entailment", "Neutral", "Contradiction"], probs)
    plt.title("NLI Softmax Probabilities")
    plt.ylabel("Probability")
    plt.show()


# Example usage
if __name__ == "__main__":
    logits = [3, 1, 0]  # Example logits
    plot_nli_probabilities(logits)
