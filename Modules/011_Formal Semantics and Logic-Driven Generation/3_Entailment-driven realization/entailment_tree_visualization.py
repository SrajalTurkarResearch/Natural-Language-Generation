# entailment_tree_visualization.py
# Purpose: Visualize an entailment tree for logical reasoning
# For aspiring scientists: Understand hierarchical entailment
# Dependencies: networkx, matplotlib (pip install networkx matplotlib)

import networkx as nx
import matplotlib.pyplot as plt


def plot_entailment_tree():
    """
    Create and display an entailment tree.
    """
    G = nx.DiGraph()
    G.add_edges_from(
        [("Canada: 10 golds", "Most golds?"), ("Most golds?", "Canada won?")]
    )
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000)
    plt.title("Entailment Tree")
    plt.show()


# Example usage
if __name__ == "__main__":
    plot_entailment_tree()
