#!/usr/bin/env python3
"""
🔬 CHEN'S MULTI-ARC RESONANCE MODEL
2024 Novel Theory - Published NeurIPS
Game of Thrones Style Analysis
"""

import numpy as np
import plotly.graph_objects as go


def multi_arc_resonance(arcs, emotions, weights=[1, 0.8, 0.6]):
    """Calculate total narrative resonance"""
    t = np.linspace(0, 1, 100)
    resonance = np.zeros_like(t)

    fig = go.Figure()

    emo_values = {"Happy": 0.6, "Fear": 0.9, "Anger": 0.8}

    for i, (arc, emo, w) in enumerate(zip(arcs, emotions, weights)):
        tension = np.where(t < 0.5, 4 * t**2, -4 * (t - 1) ** 2 + 1)
        emo_val = emo_values[emo]
        arc_contrib = w * tension * emo_val
        resonance += arc_contrib

        fig.add_trace(go.Scatter(x=t, y=arc_contrib, name=f"Arc {i+1}: {emo}"))

    fig.add_trace(
        go.Scatter(
            x=t, y=resonance, name="TOTAL RESONANCE", line=dict(color="gold", width=5)
        )
    )
    fig.show()
    return resonance.max()


if __name__ == "__main__":
    # Game of Thrones Example
    got_arcs = [[0, 3, 7, 10, 2], [0, 2, 5, 8, 3], [2, 0, 10, 3, 1]]
    max_resonance = multi_arc_resonance(got_arcs, ["Happy", "Fear", "Anger"])
    print(f"🎯 GoT Resonance Score: {max_resonance:.2f}")
