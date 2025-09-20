# amr_visualization.py
# Tutorial Component: Visualization of Abstract Meaning Representation (AMR)
# Purpose: Visualize AMR graphs to understand their structure
# Theory: AMRs are directed acyclic graphs (DAGs) with nodes (concepts) and edges (relations).
#         Visualizing AMRs helps researchers analyze meaning structures, a key skill for NLP studies.

# For Scientists: Visualization is crucial for debugging MRs and presenting research findings.
#                 This code uses networkx and matplotlib to plot AMRs, preparing you for conference papers.

# Setup Instructions:
# 1. Install required libraries: `pip install penman matplotlib networkx`
# 2. Run this file: `python amr_visualization.py`
# 3. Ensure AMR string is valid (example provided below)

import penman
from penman.model import Model
import matplotlib.pyplot as plt
import networkx as nx


# Function: Plot AMR graph
def plot_amr(amr_string):
    """
    Visualize an AMR as a directed graph.
    AMR Example: (w / want-01 :ARG0 (b / boy) :ARG1 (e / eat-01 :ARG0 b :ARG1 (p / pizza)))
    Theory: Nodes represent concepts (e.g., 'boy', 'want-01'), and edges represent relations (e.g., :ARG0).
            Visualization aids in understanding how MRs encode meaning for NLG.
    """
    # Create directed graph
    G = nx.DiGraph()
    graph = penman.decode(amr_string, model=Model())

    # Add nodes and edges
    for triple in graph.triples:
        source, role, target = triple
        source_label = source.split("/")[1] if "/" in source else source
        target_label = target.split("/")[1] if "/" in target else target
        G.add_node(source_label)
        G.add_node(target_label)
        G.add_edge(source_label, target_label, label=role)

    # Plot graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10
    )
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("AMR Graph Visualization")
    plt.show()


# Real-World Application: Visualizations are used in research to analyze MR structures (e.g., in ACL papers).
# Research Direction: Develop tools to visualize multimodal MRs (text + images).
# Tip for Scientists: Save visualizations as PDFs for research reports or presentations.

# Test the visualization
if __name__ == "__main__":
    # Example AMR
    amr = "(w / want-01 :ARG0 (b / boy) :ARG1 (e / eat-01 :ARG0 b :ARG1 (p / pizza)))"
    plot_amr(amr)

    # For Your Notebook: Sketch the AMR graph manually and compare with the plotted output.
    #                   Try visualizing a new AMR, e.g., for "John loves Mary."

# Future Direction: Integrate visualizations with interactive tools (e.g., Jupyter widgets).
# Missing from Tutorial: Dynamic visualizations for real-time AMR parsing.
# Next Steps: Explore graph visualization libraries like Plotly for interactive AMR displays.
