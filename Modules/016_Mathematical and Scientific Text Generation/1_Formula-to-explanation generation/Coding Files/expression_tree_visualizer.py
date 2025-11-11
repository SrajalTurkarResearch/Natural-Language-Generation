# expression_tree_visualizer.py
# Author: Grok (inspired by Turing, Einstein, Tesla)
# Description: Standalone script to visualize a mathematical expression tree.
# Usage: python expression_tree_visualizer.py
# Dependencies: matplotlib, networkx (pip install matplotlib networkx)

import matplotlib.pyplot as plt
import networkx as nx


def visualize_expression_tree():
    """
    Create and display a graph visualization of a sample mathematical expression tree.
    Example: a + b * (c - d)
    """
    # Step 1: Create a directed graph for the tree
    G = nx.DiGraph()

    # Step 2: Add nodes and edges for the expression (hierarchical structure)
    G.add_edges_from(
        [
            ("+", "a"),  # Left operand of addition
            ("+", "*"),  # Right operand is multiplication
            ("*", "b"),  # Left of multiplication
            ("*", "-"),  # Right is subtraction
            ("-", "c"),  # Left of subtraction
            ("-", "d"),  # Right of subtraction
        ]
    )

    # Step 3: Layout and draw the graph
    pos = nx.spring_layout(G)  # Automatic positioning
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=12,
        font_weight="bold",
    )
    plt.title("Expression Tree: a + b * (c - d)")
    plt.show()  # Displays the plot window


# Run the visualization
if __name__ == "__main__":
    visualize_expression_tree()
