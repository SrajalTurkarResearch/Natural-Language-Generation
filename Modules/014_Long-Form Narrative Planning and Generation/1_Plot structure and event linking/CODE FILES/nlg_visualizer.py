#!/usr/bin/env python3
"""
📊 VISUALIZER: Graphs, Plots, Dashboards
Scientist-level visualizations
"""

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np


class Visualizer:
    @staticmethod
    def visualize_narrative(graph: nx.DiGraph):
        """Interactive narrative graph"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, seed=42)

        # Nodes
        nx.draw_networkx_nodes(
            graph, pos, node_color="lightblue", node_size=2000, alpha=0.9
        )

        # Weighted edges
        for u, v, data in graph.edges(data=True):
            nx.draw_networkx_edges(
                graph, pos, edgelist=[(u, v)], width=data["weight"] * 5, alpha=0.7
            )

        # Labels
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")
        edge_labels = {
            (u, v): f'{d["weight"]:.2f}' for u, v, d in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)

        plt.title("🕸️ NARRATIVE GRAPH: Events & Causal Links")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def coherence_dashboard(graph: nx.DiGraph):
        """Analyze 10 random narratives"""
        paths = []
        scores = []

        nodes = list(graph.nodes)
        for _ in range(10):
            try:
                path = nx.shortest_path(
                    graph, random.choice(nodes), random.choice(nodes)
                )
                if len(path) > 1:
                    prob = np.prod(
                        [
                            graph[path[i]][path[i + 1]]["weight"]
                            for i in range(len(path) - 1)
                        ]
                    )
                    paths.append(" → ".join(path))
                    scores.append(prob)
            except:
                continue

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(scores, ax=ax1, bins=5, kde=True, color="skyblue")
        ax1.set_title("📈 Coherence Distribution")
        sns.scatterplot(x=[len(p.split(" → ")) for p in paths], y=scores, ax=ax2, s=100)
        ax2.set_title("📏 Length vs Quality")
        plt.tight_layout()
        plt.show()

        best_idx = np.argmax(scores)
        print(f"🏆 BEST: {paths[best_idx]} (Score: {scores[best_idx]:.3f})")
