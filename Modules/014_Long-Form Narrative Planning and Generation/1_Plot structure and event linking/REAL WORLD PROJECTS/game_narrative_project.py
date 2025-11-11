#!/usr/bin/env python3
"""
🎮 PROJECT: Game Narrative NLG
Dynamic stories for video games (Hero's Journey)
Author: Grok (xAI) | Date: Oct 17, 2025
Run: python game_narrative_project.py
Dependencies: nlg_core.py, nlg_visualizer.py
"""

import random
import networkx as nx
import pandas as pd
from typing import Dict, List
from nlg_core import NLGCore
from nlg_visualizer import Visualizer


class GameNarrativeProject:
    def __init__(self):
        """Initialize with game-style events"""
        self.nlg = NLGCore()
        self.dataset = self._load_simulated_data()
        self.graph = self._build_game_graph()

    def _load_simulated_data(self) -> Dict[str, List[str]]:
        """Simulated game events (mimics RPG quest data)"""
        return {
            "exposition": [
                "Hero discovers ancient prophecy",
                "Villager seeks hero's help",
            ],
            "rising_action": [
                "Hero finds magic sword",
                "Hero trains with elder",
                "Hero crosses dark forest",
            ],
            "climax": ["Hero battles dark sorcerer", "Hero solves ancient puzzle"],
            "falling_action": ["Sorcerer defeated", "Kingdom restored"],
            "resolution": ["Hero returns as legend", "Village celebrates victory"],
        }

    def _build_game_graph(self) -> nx.DiGraph:
        """G = (V, E) for game narrative (Hero's Journey)"""
        G = nx.DiGraph()
        stages = [
            "exposition",
            "rising_action",
            "climax",
            "falling_action",
            "resolution",
        ]

        # Add nodes
        for stage, events in self.dataset.items():
            for event in events:
                G.add_node(event, stage=stage)

        # Add branching edges (game-like choices)
        edges = []
        for i in range(len(stages) - 1):
            for e1 in self.dataset[stages[i]]:
                for e2 in self.dataset[stages[i + 1]]:
                    weight = random.uniform(0.65, 0.95)  # Branching uncertainty
                    edges.append((e1, e2, weight))

        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        return G

    def generate_game_narrative(self, player_choice: str = None) -> str:
        """Generate dynamic game story"""
        story = []
        current_stage = "exposition"
        transitions = ["", "NEXT ", "EPIC MOMENT: ", "THEN ", "VICTORY: "]

        while current_stage:
            stage_events = [
                n for n, d in self.graph.nodes(data=True) if d["stage"] == current_stage
            ]
            if player_choice and current_stage == "rising_action":
                event = next(
                    (e for e in stage_events if player_choice.lower() in e.lower()),
                    random.choice(stage_events),
                )
            else:
                event = random.choice(stage_events)
            story.append(event)
            current_stage = self.nlg.links.get(current_stage)

        narrative = [f"{t}{e}" for t, e in zip(transitions, story)]
        coherence = self.nlg.calculate_coherence()
        return f"🎮 GAME NARRATIVE\n{'='*30}\n{' | '.join(narrative)}\n\n📊 Coherence: {coherence:.3f}"

    def visualize(self):
        """Visualize game narrative graph"""
        Visualizer.visualize_narrative(self.graph)

    def analyze_coherence(self) -> pd.DataFrame:
        """Analyze narrative paths"""
        paths = []
        scores = []
        for _ in range(5):
            try:
                start = random.choice(self.dataset["exposition"])
                end = random.choice(self.dataset["resolution"])
                path = nx.shortest_path(self.graph, start, end)
                score = self.nlg.generate_path_probability(path)
                paths.append(" → ".join(path))
                scores.append(score)
            except:
                continue

        return pd.DataFrame({"Path": paths, "Coherence Score": scores})


def main():
    print("🎮 STARTING GAME NARRATIVE PROJECT")
    project = GameNarrativeProject()

    # Generate narrative with player choice
    print("\n📜 DEFAULT NARRATIVE:")
    print(project.generate_game_narrative())

    print("\n📜 PLAYER-DRIVEN NARRATIVE (chose 'sword'):")
    print(project.generate_game_narrative(player_choice="sword"))

    # Visualize
    project.visualize()

    # Analyze
    df = project.analyze_coherence()
    print("\n📈 COHERENCE ANALYSIS:")
    print(df)


if __name__ == "__main__":
    main()
