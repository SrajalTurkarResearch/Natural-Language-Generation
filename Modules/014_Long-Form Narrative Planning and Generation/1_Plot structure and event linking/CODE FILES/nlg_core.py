#!/usr/bin/env python3
"""
🔬 NLG CORE: Plot Structure + Event Linking Engine
Mathematical foundation for scientist-level NLG
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List


class NLGCore:
    def __init__(self):
        self.graph = self._build_narrative_graph()
        self.events_db = self._load_events()
        self.links = self._load_links()

    def _build_narrative_graph(self) -> nx.DiGraph:
        """G = (V, E) - Mathematical narrative graph"""
        G = nx.DiGraph()

        # Events (V = nodes)
        events = [
            "Find map",
            "Explore cave",
            "Fight dragon",
            "Get treasure",
            "Return home",
        ]
        G.add_nodes_from(events)

        # Causal links (E = weighted edges)
        edges = [
            ("Find map", "Explore cave", 0.95),
            ("Explore cave", "Fight dragon", 0.85),
            ("Fight dragon", "Get treasure", 0.90),
            ("Get treasure", "Return home", 0.98),
        ]

        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    def _load_events(self) -> Dict[str, List[str]]:
        """Freytag's Pyramid events"""
        return {
            "exposition": ["Alice found a magic book", "Bob got mysterious letter"],
            "rising_action": ["She read about hidden cave", "He traveled dark forest"],
            "climax": ["Alice faced dragon", "Bob solved final riddle"],
            "falling_action": ["She escaped with treasure", "He returned safely"],
            "resolution": ["Alice became hero", "Bob shared wisdom"],
        }

    def _load_links(self) -> Dict:
        """Causal progression"""
        return {
            "exposition": "rising_action",
            "rising_action": "climax",
            "climax": "falling_action",
            "falling_action": "resolution",
        }

    def generate_complete_story(self) -> str:
        """Generate full Freytag's Pyramid story"""
        story = []
        stage = "exposition"

        while stage:
            event = random.choice(self.events_db[stage])
            story.append(event)
            stage = self.links.get(stage)

        transitions = ["", "→", "SUDDENLY →", "THEN →", "FINALLY:"]
        return " ".join([f"{t} {e}" for t, e in zip(transitions, story)])

    def calculate_coherence(self) -> float:
        """C = Σw(e)/|E| - Mathematical coherence"""
        weights = [d["weight"] for _, _, d in self.graph.edges(data=True)]
        return sum(weights) / len(weights)

    def generate_path_probability(self, path: List[str]) -> float:
        """P(path) = ∏P(v_{i+1}|v_i)"""
        prob = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                prob *= self.graph[u][v]["weight"]
        return prob
