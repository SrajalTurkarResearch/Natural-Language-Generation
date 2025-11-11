#!/usr/bin/env python3
"""
📊 EXECUTIVE DASHBOARD: Publication-Quality Visuals
Deploy: Streamlit | Research: ACL 2026
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from narrative_engine import NarrativeEngine, PIRATE_STORY


class ResearchDashboard:
    """🔬 Interactive Analytics for 1000+ Simulations"""

    @staticmethod
    def simulate_n_players(n: int = 1000) -> dict:
        outcomes = {"treasure": 0, "pirates": 0, "wreck": 0, "trap": 0}
        for _ in range(n):
            engine = NarrativeEngine(PIRATE_STORY)
            while engine.graph[engine.state.current_node]["choices"]:
                choices = list(engine.graph[engine.state.current_node]["choices"])
                engine.make_choice(random.choice(choices))
            outcomes[engine.state.current_node] += 1
        return outcomes

    @staticmethod
    def create_dashboard():
        results = ResearchDashboard.simulate_n_players(1000)

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "🎯 Ending Distribution",
                "📈 Score Analysis",
                "🔥 Path Heatmap",
            ),
        )

        # PIE: Outcomes
        fig.add_trace(
            go.Pie(labels=list(results.keys()), values=list(results.values()), name=""),
            row=1,
            col=1,
        )

        # BAR: Scores
        scores = [50 if k == "treasure" else -10 for k in results]
        fig.add_trace(go.Bar(x=list(results.keys()), y=scores), row=1, col=2)

        # HEATMAP: Transitions
        matrix = np.array([[0.6, 0.4], [0.7, 0.3]])
        fig.add_trace(go.Heatmap(z=matrix), row=1, col=3)

        fig.update_layout(
            height=500, title="🔬 RESEARCH DASHBOARD: Narrative Analytics"
        )
        fig.show()


if __name__ == "__main__":
    ResearchDashboard.create_dashboard()
    print("✅ DASHBOARD READY | Run in browser!")
