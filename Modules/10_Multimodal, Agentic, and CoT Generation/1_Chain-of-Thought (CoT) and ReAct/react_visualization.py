# react_visualization.py
# Purpose: Visualize the ReAct loop as a flowchart using matplotlib.
# Context: Illustrates the iterative cycle of Reason → Act → Observe, key for dynamic NLG.
# For scientists: Visualizing cycles aids in designing iterative experiments.

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def plot_react_loop():
    """
    Creates a flowchart for ReAct: Reason → Act → Observe → Repeat.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # Reason box
    ax.add_patch(Rectangle((0.4, 0.8), 0.2, 0.1, fill=None))
    ax.text(0.5, 0.85, "Reason", ha="center")
    # Arrow to Act
    ax.add_patch(
        FancyArrowPatch(
            (0.6, 0.85), (0.8, 0.65), arrowstyle="->", connectionstyle="arc3,rad=0.3"
        )
    )
    # Act box
    ax.add_patch(Rectangle((0.8, 0.4), 0.2, 0.1, fill=None))
    ax.text(0.9, 0.45, "Act", ha="center")
    # Arrow to Observe
    ax.add_patch(
        FancyArrowPatch(
            (0.9, 0.4), (0.7, 0.15), arrowstyle="->", connectionstyle="arc3,rad=0.3"
        )
    )
    # Observe box
    ax.add_patch(Rectangle((0.4, 0.0), 0.2, 0.1, fill=None))
    ax.text(0.5, 0.05, "Observe", ha="center")
    # Arrow to Repeat
    ax.add_patch(
        FancyArrowPatch(
            (0.4, 0.05), (0.2, 0.25), arrowstyle="->", connectionstyle="arc3,rad=0.3"
        )
    )
    # Repeat box
    ax.add_patch(Rectangle((0.0, 0.4), 0.2, 0.1, fill=None))
    ax.text(0.1, 0.45, "Repeat", ha="center")
    # Arrow back to Reason
    ax.add_patch(
        FancyArrowPatch(
            (0.1, 0.5), (0.3, 0.75), arrowstyle="->", connectionstyle="arc3,rad=0.3"
        )
    )
    ax.axis("off")
    plt.title("ReAct Loop Flowchart")
    plt.show()


if __name__ == "__main__":
    print("Generating ReAct Loop Visualization")
    plot_react_loop()

# Explanation for Aspiring Scientists:
# - This shows ReAct's cyclic process, like iterating in a lab experiment.
# - In NLG, it ensures generated text is grounded in verified data.
# - Experiment by adding labels for specific actions (e.g., 'Query API').
