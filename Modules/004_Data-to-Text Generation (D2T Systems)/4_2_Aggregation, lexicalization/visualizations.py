# Visualizations for Aggregation and Lexicalization in NLG
#
# Theory:
# - Visualizations help understand how aggregation and lexicalization transform data into text.
# - Aggregation Visualization: Shows how multiple data points combine into one sentence (e.g., flowchart).
# - Lexicalization Visualization: Illustrates mapping of data to words (e.g., temperature to 'warm').
# - Purpose: Clarify processes for learning and research.
# - Analogy: Visualizations are like blueprints, showing how raw materials (data) become a finished product (text).
#
# This file includes two visualizations:
# 1. Aggregation Flowchart: Combines weather data points.
# 2. Lexicalization Bar Plot: Maps temperatures to descriptive words.
#
# Requirements: Install matplotlib (`pip install matplotlib`).
# For your research: Use visualizations to debug NLG pipelines and present findings.

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


def plot_aggregation_flowchart():
    """
    Plots a flowchart showing aggregation of weather data.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data points
    ax.add_patch(Rectangle((0.1, 0.8), 0.2, 0.1, fill=True, color="lightblue"))
    ax.text(0.2, 0.85, "Temp: 22°C", ha="center")
    ax.add_patch(Rectangle((0.4, 0.8), 0.2, 0.1, fill=True, color="lightblue"))
    ax.text(0.5, 0.85, "Sky: Sunny", ha="center")

    # Arrows to aggregation
    ax.add_patch(FancyArrowPatch((0.2, 0.8), (0.35, 0.6), mutation_scale=20))
    ax.add_patch(FancyArrowPatch((0.5, 0.8), (0.35, 0.6), mutation_scale=20))

    # Aggregation process
    ax.add_patch(Rectangle((0.3, 0.5), 0.2, 0.1, fill=True, color="lightgreen"))
    ax.text(0.4, 0.55, "Aggregate", ha="center")

    # Arrow to output
    ax.add_patch(FancyArrowPatch((0.4, 0.5), (0.4, 0.3), mutation_scale=20))

    # Output
    ax.add_patch(Rectangle((0.1, 0.2), 0.6, 0.1, fill=True, color="lightcoral"))
    ax.text(0.4, 0.25, "It’s sunny with 22°C", ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.title("Aggregation Flowchart")
    plt.show()


def lexicalize_temperature(temp):
    """Helper function for lexicalization visualization."""
    if temp < 10:
        return "cold"
    elif 10 <= temp <= 20:
        return "cool"
    elif 21 <= temp <= 27:
        return "warm"
    else:
        return "hot"


def plot_lexicalization_mapping():
    """
    Plots a bar chart mapping temperatures to descriptive words.
    """
    temps = np.arange(0, 35, 5)
    descriptions = [lexicalize_temperature(t) for t in temps]

    plt.figure(figsize=(8, 4))
    plt.bar(temps, descriptions, color="skyblue")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Description")
    plt.title("Lexicalization: Temperature Mapping")
    plt.show()


# Example usage
if __name__ == "__main__":
    plot_aggregation_flowchart()
    plot_lexicalization_mapping()

    # Research Tip: Create visualizations for other NLG processes (e.g., sentence planning).
