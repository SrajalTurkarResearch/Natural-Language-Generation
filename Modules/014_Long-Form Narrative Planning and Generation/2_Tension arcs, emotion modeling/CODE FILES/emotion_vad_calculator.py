#!/usr/bin/env python3
"""
🧠 VAD EMOTION CALCULATOR
3D Emotion Space + Distance Metrics
"""

import numpy as np
import plotly.graph_objects as go


class Emotion:
    def __init__(self, name, v, a, d):
        self.name = name
        self.v, self.a, self.d = v, a, d

    def distance(self, other):
        return np.sqrt(
            (self.v - other.v) ** 2 + (self.a - other.a) ** 2 + (self.d - other.d) ** 2
        )


# STANDARD EMOTIONS DATABASE
EMOTIONS = {
    "Happy": Emotion("Happy", 0.8, 0.6, 0.6),
    "Sad": Emotion("Sad", -0.8, 0.2, -0.6),
    "Fear": Emotion("Fear", -0.7, 0.9, -0.8),
    "Anger": Emotion("Anger", -0.6, 0.8, 0.7),
    "Surprise": Emotion("Surprise", 0.2, 0.9, 0.0),
}


def visualize_vad_space():
    """3D Emotion Space Visualization"""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[e.v for e in EMOTIONS.values()],
                y=[e.a for e in EMOTIONS.values()],
                z=[e.d for e in EMOTIONS.values()],
                mode="markers+text",
                marker=dict(size=10, color="red"),
                text=list(EMOTIONS.keys()),
                textposition="middle center",
            )
        ]
    )

    fig.update_layout(
        title="3D VAD Emotion Space",
        scene=dict(
            xaxis_title="Valence", yaxis_title="Arousal", zaxis_title="Dominance"
        ),
    )
    fig.show()


def calculate_distances():
    """Print all emotion distances"""
    print("📏 EMOTION DISTANCES:")
    for e1 in EMOTIONS:
        for e2 in EMOTIONS:
            if e1 != e2:
                dist = EMOTIONS[e1].distance(EMOTIONS[e2])
                print(f"{e1} → {e2}: {dist:.2f}")


if __name__ == "__main__":
    visualize_vad_space()
    calculate_distances()
