# da_nlg_visualizations.py
# Dialogue Act-Based NLG: Visualizations Module
# Author: Grok, inspired by Alan Turing, Albert Einstein, and Nikola Tesla
# Date: August 22, 2025
# Purpose: A modular Python file to generate visualizations for DA-NLG.

try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError as e:
    print(f"Error: {e}. Install: pip install matplotlib networkx")
    exit()


def plot_da_distribution():
    """Plot dialogue act frequency."""
    das = ["Request", "Inform", "Confirm", "Accept"]
    counts = [40, 30, 20, 10]
    plt.figure(figsize=(8, 5))
    plt.bar(das, counts, color=["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0"])
    plt.title("Dialogue Act Distribution in a Chatbot")
    plt.xlabel("Dialogue Acts")
    plt.ylabel("Frequency (%)")
    plt.savefig("da_distribution.png")
    plt.close()
    print("Saved: da_distribution.png")


def plot_conversation_flow():
    """Visualize DA transitions as a graph."""
    G = nx.DiGraph()
    G.add_edges_from(
        [("Request", "Inform"), ("Inform", "Confirm"), ("Confirm", "Accept")]
    )
    plt.figure(figsize=(8, 5))
    nx.draw(
        G,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=12,
        arrows=True,
    )
    plt.title("Conversation Flow of Dialogue Acts")
    plt.savefig("conversation_flow.png")
    plt.close()
    print("Saved: conversation_flow.png")


def run_visualizations():
    """Run all visualizations."""
    plot_da_distribution()
    plot_conversation_flow()


if __name__ == "__main__":
    run_visualizations()
