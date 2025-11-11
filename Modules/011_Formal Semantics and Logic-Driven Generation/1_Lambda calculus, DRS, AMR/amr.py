# amr.py
# Implements Abstract Meaning Representation (AMR) for NLG tutorial
# Designed for aspiring scientists, with visualizations
# Requires: networkx, matplotlib
# Run: python amr.py

import networkx as nx
import matplotlib.pyplot as plt

# --- AMR Basics ---
# AMR represents sentence meanings as graphs, ignoring grammar.
# Used in NLG for abstract, language-neutral meanings, e.g., "Fox jumps over dog."
# Components: Nodes (concepts like jump-01), Edges (relations like :ARG0).

# Simple AMR Graph: "Quick brown fox jumps over lazy dog"
G = nx.DiGraph()
G.add_nodes_from(["jump-01", "fox", "dog", "brown", "quick", "lazy"])
G.add_edges_from(
    [
        ("jump-01", "fox", {"label": ":ARG0"}),
        ("jump-01", "dog", {"label": ":ARG1"}),
        ("fox", "brown", {"label": ":mod"}),
        ("fox", "quick", {"label": ":mod"}),
        ("dog", "lazy", {"label": ":mod"}),
    ]
)

# Visualize AMR
print("Generating AMR Graph...")
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.title("AMR: Quick brown fox jumps over lazy dog")
plt.show()

# --- Exercise ---
# Parse "E=mc²" into AMR.
# Answer: (e / equal-01 :ARG1 (e2 / energy) :ARG2 (p / product-01 :ARG1 (m / mass) :ARG2 (p2 / product-01 :ARG1 (c / c) :op2 2)))
G_exercise = nx.DiGraph()
G_exercise.add_nodes_from(["equal-01", "energy", "product-01", "mass", "c"])
G_exercise.add_edges_from(
    [
        ("equal-01", "energy", {"label": ":ARG1"}),
        ("equal-01", "product-01", {"label": ":ARG2"}),
        ("product-01", "mass", {"label": ":ARG1"}),
        ("product-01", "c", {"label": ":ARG2"}),
        ("c", "2", {"label": ":op2"}),
    ]
)

print("\nExercise AMR Graph...")
pos = nx.spring_layout(G_exercise)
nx.draw(G_exercise, pos, with_labels=True, node_color="lightgreen", node_size=2000)
edge_labels = nx.get_edge_attributes(G_exercise, "label")
nx.draw_networkx_edge_labels(G_exercise, pos, edge_labels)
plt.title("AMR: E=mc²")
plt.show()

# --- 2025 Update ---
# Neural AMR (2025) uses graph networks for better parsing in LLMs.
# AMR-DA augments data for varied NLG outputs.
# Try: Augment "Cat sleeps" AMR with :mod (color black).

# --- For Scientists ---
# AMR’s abstraction aids multilingual science reports, like chemistry reactions.
# Use in physics: Structure relativity equations for NLG.
