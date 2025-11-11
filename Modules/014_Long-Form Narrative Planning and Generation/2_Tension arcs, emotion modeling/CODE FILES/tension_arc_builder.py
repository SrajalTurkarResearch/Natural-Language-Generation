#!/usr/bin/env python3
"""
🌟 TENSION ARC BUILDER
Interactive tool to visualize Freytag's Pyramid
Author: Dr. Alex Chen | MIT ACL Fellow 2023
"""

import numpy as np
import plotly.graph_objects as go
import argparse


def plot_tension_arc(a=4, segments=5, title="My Story"):
    """Generate interactive tension arc visualization"""
    t = np.linspace(0, 1, segments)
    tension = np.where(t < 0.5, a * t**2, -a * (t - 1) ** 2 + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=tension,
            mode="lines+markers",
            name="Tension",
            line=dict(color="red", width=4),
        )
    )

    fig.update_layout(
        title=f"Tension Arc: {title}",
        xaxis_title="Story Progress",
        yaxis_title="Tension (0-1)",
        height=400,
        template="plotly_dark",
    )
    fig.show()
    return tension


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steepness", type=int, default=4)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--title", default="My Story")
    args = parser.parse_args()

    print("🚀 Building Tension Arc...")
    plot_tension_arc(args.steepness, args.segments, args.title)
