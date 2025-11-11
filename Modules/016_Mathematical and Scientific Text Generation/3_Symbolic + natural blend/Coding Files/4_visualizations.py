# 4_visualizations.py
# Visual tools: Knowledge graphs and probability plots

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_knowledge_graph():
    """Draw a symbolic knowledge graph for weather concepts."""
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("Weather", "Temperature"),
            ("Weather", "Condition"),
            ("Temperature", "Warm"),
            ("Temperature", "Cool"),
            ("Condition", "Sunny"),
            ("Condition", "Rainy"),
        ]
    )
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        font_size=12,
        font_weight="bold",
        arrows=True,
    )
    plt.title("Symbolic Knowledge Graph for Weather NLG")
    plt.show()


def plot_word_probabilities():
    """Simulate neural word selection with softmax."""
    words = ["mat", "dog", "hat"]
    logits = [3.0, 1.0, 0.5]  # Raw neural scores
    probs = np.exp(logits) / np.sum(np.exp(logits))

    plt.bar(words, probs, color=["orange", "lightblue", "lightgreen"])
    plt.title("Neural Model: Word Selection Probabilities")
    plt.ylabel("Probability")
    for i, p in enumerate(probs):
        plt.text(i, p + 0.01, f"{p:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.show()


# === RUN ===
if __name__ == "__main__":
    plot_knowledge_graph()
    plot_word_probabilities()
