# visualizer.py
# Author: Grok (Tesla's visionary schematics)
# Purpose: Generate visualizations for NLG ethics.
# Usage: from visualizer import plot_bias_bars

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_bias_bars(groups, pre_mitigation, post_mitigation, title="Bias Mitigation"):
    """
    Bar plot for bias rates pre/post mitigation.

    Parameters:
    - groups (list): Labels e.g., ['Male', 'Female'].
    - pre_mitigation (list): Rates before.
    - post_mitigation (list): Rates after.
    - title (str): Plot title.

    Returns:
    - None (shows plot).

    Logic: Uneven bars indicate bias; aim for parity.
    """
    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, pre_mitigation, width, label="Pre")
    ax.bar(x + width / 2, post_mitigation, width, label="Post")
    ax.set_ylabel("Positive Output Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    plt.show()


def plot_attention_heatmap(attention_matrix, title="Attention Map"):
    """
    Heatmap for model explainability.

    Parameters:
    - attention_matrix (np.array): 2D attention weights.
    - title (str): Plot title.
    """
    sns.heatmap(attention_matrix, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.show()


# Example:
if __name__ == "__main__":
    plot_bias_bars(["Male", "Female"], [0.8, 0.6], [0.7, 0.7])
