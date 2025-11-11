# visualizations.py
# Generate visualizations for NLG concepts.
# Author: Visualizing like Richard Feynman's diagrams.
# Requirements: matplotlib, networkx, numpy.
# Usage: Run to display plots.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_grammar_tree():
    """Symbolic grammar tree diagram."""
    G = nx.DiGraph()
    G.add_edges_from(
        [("S", "NP"), ("S", "VP"), ("NP", "Det"), ("NP", "N"), ("VP", "V")]
    )
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue")
    plt.title("Symbolic Grammar Tree")
    plt.show()


def visualize_attention_heatmap():
    """Neural attention heatmap (simplified)."""
    attn = np.random.rand(5, 5)
    plt.imshow(attn, cmap="hot")
    plt.title("Neural Attention Heatmap")
    plt.show()


def visualize_neurosymbolic_arch():
    """Hybrid architecture flow."""
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("Data", "Neural"),
            ("Data", "Symbolic"),
            ("Neural", "Output"),
            ("Symbolic", "Output"),
        ]
    )
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightgreen")
    plt.title("Neurosymbolic Architecture")
    plt.show()


if __name__ == "__main__":
    # Run visualizations sequentially, like lab demos.
    visualize_grammar_tree()
    visualize_attention_heatmap()
    visualize_neurosymbolic_arch()
    # Experiment: Modify graphs to represent your own NLG ideas.
