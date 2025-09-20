# cot_visualization.py
# Purpose: Visualize the CoT process as a flowchart using matplotlib.
# Context: Helps understand CoT's linear reasoning structure, useful for NLG transparency.
# For scientists: Visualizations are key to communicating complex processes clearly.

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def plot_cot_flowchart():
    """
    Creates a simple flowchart for CoT: Question → Steps → Answer.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    # Question box
    ax.add_patch(Rectangle((0.1, 0.8), 0.8, 0.1, fill=None, edgecolor="black"))
    ax.text(0.5, 0.85, "Question", ha="center")
    # Arrow to first step
    ax.add_patch(FancyArrowPatch((0.5, 0.8), (0.5, 0.7), arrowstyle="->"))
    # Step 1 box
    ax.add_patch(Rectangle((0.1, 0.5), 0.8, 0.1, fill=None, edgecolor="black"))
    ax.text(0.5, 0.55, "Step 1: Decompose", ha="center")
    # Arrow to final answer
    ax.add_patch(FancyArrowPatch((0.5, 0.5), (0.5, 0.4), arrowstyle="->"))
    # Answer box
    ax.add_patch(Rectangle((0.1, 0.2), 0.8, 0.1, fill=None, edgecolor="black"))
    ax.text(0.5, 0.25, "Final Answer", ha="center")
    ax.axis("off")
    plt.title("Chain-of-Thought Flowchart")
    plt.show()


if __name__ == "__main__":
    print("Generating CoT Flowchart Visualization")
    plot_cot_flowchart()

# Explanation for Aspiring Scientists:
# - This diagram shows CoT's step-by-step flow, like a lab protocol.
# - In NLG, such visuals clarify how AI generates structured text.
# - Try customizing: add more steps or labels for specific problems.
