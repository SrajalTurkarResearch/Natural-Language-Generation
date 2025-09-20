# visualizations.py
# Visualizations for Multi-Agent NLG system
# Purpose: Create diagrams and plots to illustrate agent interactions and performance
# Author: Inspired by Turing, Einstein, and Tesla for aspiring scientists
# Prerequisites: Python 3.x, matplotlib, networkx

"""
This script generates visualizations for the Multi-Agent NLG system:
- A graph showing agent interactions.
- A plot of agent performance over iterations.
Run this script to visualize the system's structure and learning curve.
"""

import networkx as nx
import matplotlib.pyplot as plt


def plot_agent_architecture():
    """
    Plot a directed graph of agent interactions.
    """
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("Data Agent", "Planning Agent"),
            ("Planning Agent", "Generation Agent"),
            ("Generation Agent", "Refinement Agent"),
        ]
    )
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        font_weight="bold",
        node_size=2000,
        arrowsize=20,
    )
    plt.title("Multi-Agent NLG Architecture")
    plt.show()


def plot_learning_curve():
    """
    Plot the learning agent's utility over iterations.
    """
    iterations = range(10)
    utility = [0.5 + 0.05 * i for i in iterations]
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, utility, marker="o")
    plt.xlabel("Iterations")
    plt.ylabel("Utility Score")
    plt.title("Agent Learning Curve")
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to generate visualizations.
    """
    print("Generating Agent Architecture Diagram...")
    plot_agent_architecture()
    print("Generating Learning Curve Plot...")
    plot_learning_curve()


if __name__ == "__main__":
    main()

# Try This: Modify the graph edges or utility formula to experiment with visualizations.
# Next Steps: Add labels to edges or plot multiple agents' performance.
