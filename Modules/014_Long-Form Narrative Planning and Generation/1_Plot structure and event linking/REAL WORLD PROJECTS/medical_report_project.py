#!/usr/bin/env python3
"""
🏥 PROJECT: Medical Report NLG
Automated patient narratives for hospitals
Author: Grok (xAI) | Date: Oct 17, 2025
Run: python medical_report_project.py
Dependencies: nlg_core.py, nlg_visualizer.py
"""

import random
import networkx as nx
import pandas as pd
from typing import Dict, List
from nlg_core import NLGCore
from nlg_visualizer import Visualizer


class MedicalReportProject:
    def __init__(self):
        """Initialize with simulated patient data"""
        self.nlg = NLGCore()
        self.dataset = self._load_simulated_data()
        self.graph = self._build_medical_graph()

    def _load_simulated_data(self) -> Dict[str, List[str]]:
        """Simulated patient data (mimics EHR systems)"""
        return {
            "exposition": [
                "Patient: John Doe, 45yo, admitted",
                "Patient: Jane Smith, 32yo, ER visit",
            ],
            "rising_action": [
                "Reports chest pain, shortness of breath",
                "Fever and cough noted",
                "Blood pressure elevated",
            ],
            "climax": ["ECG reveals coronary blockage", "X-ray shows pneumonia"],
            "falling_action": [
                "Angioplasty performed successfully",
                "Antibiotics administered",
                "Patient stabilized in ICU",
            ],
            "resolution": ["Discharged with follow-up plan", "Full recovery expected"],
        }

    def _build_medical_graph(self) -> nx.DiGraph:
        """G = (V, E) for medical narrative"""
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

        # Add causal edges
        edges = []
        for i in range(len(stages) - 1):
            for e1 in self.dataset[stages[i]]:
                for e2 in self.dataset[stages[i + 1]]:
                    weight = random.uniform(0.75, 0.98)  # High medical causality
                    edges.append((e1, e2, weight))

        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        return G

    def generate_medical_report(self) -> str:
        """Generate hospital-grade report"""
        story = []
        current_stage = "exposition"

        while current_stage:
            stage_events = [
                n for n, d in self.graph.nodes(data=True) if d["stage"] == current_stage
            ]
            event = random.choice(stage_events)
            story.append(event)
            current_stage = self.nlg.links.get(current_stage)

        report = "🏥 MEDICAL REPORT\n" + "=" * 30 + "\n"
        for i, event in enumerate(story):
            report += f"Stage {i+1}: {event}\n"

        coherence = self.nlg.calculate_coherence()
        return f"{report}\n📊 Coherence: {coherence:.3f}"

    def visualize(self):
        """Visualize medical narrative graph"""
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
    print("🏥 STARTING MEDICAL REPORT PROJECT")
    project = MedicalReportProject()

    # Generate report
    print(project.generate_medical_report())

    # Visualize
    project.visualize()

    # Analyze
    df = project.analyze_coherence()
    print("\n📈 COHERENCE ANALYSIS:")
    print(df)


if __name__ == "__main__":
    main()
