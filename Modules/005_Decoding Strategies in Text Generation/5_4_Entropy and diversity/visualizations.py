# visualizations.py
# Theory: Visuals make abstract math tangible—plot probability distributions to see entropy's 'spread'.
# High entropy: Even bars (diverse); Low: Skewed (predictable).
# Analogy: Tesla's AC current waves—entropy as information 'waveform'.
# Rare Insight: In NLG research (e.g., EACL 2021), visualize n-gram entropies to debug diversity issues.

import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(probs, title):
    """
    Plot bar graph of probabilities.
    Args: probs (list), title (str)
    """
    labels = [f"Outcome {i+1}" for i in range(len(probs))]
    plt.bar(labels, probs)
    plt.title(title)
    plt.xlabel("Outcomes")
    plt.ylabel("Probability")
    plt.show()


# Low Entropy Example
probs_low = [0.99, 0.01]
plot_distribution(probs_low, "Low Entropy Distribution (Predictable)")

# High Entropy Example
probs_high = [0.25] * 4
plot_distribution(probs_high, "High Entropy Distribution (Diverse)")

# NLG Visualization: Word freq in repetitive vs. diverse text
# Real-world: Analyze scientific abstracts for diversity (e.g., Einstein's papers vs. repetitive AI summaries).
word_probs_repetitive = [0.25, 0.75]  # "The cat cat cat"
plot_distribution(word_probs_repetitive, "Repetitive Text (Low Entropy)")

# Tip: Use plt.savefig('fig.png') for research papers.
# Next Steps: Animate entropy changes over training epochs in ML models.
