# visualization_dataset.py
# Visualizations and dataset processing for logic-to-text mapping in NLG
# Purpose: Create logic trees, flowcharts, and process Logic2Text dataset examples
# For aspiring scientists: Visualize like Einstein, experiment like Tesla with real data

# Dependencies: Install with pip
# pip install graphviz datasets matplotlib
# Also install system graphviz (e.g., sudo apt install graphviz on Linux)

import graphviz
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from datasets import load_dataset


def create_logic_tree():
    """
    Create a logic tree visualization using Graphviz.
    Example: AND(Eats(John, Apple), Likes(John, Fruit))
    """
    dot = graphviz.Digraph(comment="Logic Tree")
    dot.node("A", "AND")
    dot.node("B", "Eats(John, Apple)")
    dot.node("C", "Likes(John, Fruit)")
    dot.edges(["AB", "AC"])
    dot.render("logic_tree", format="png", view=False)  # Saves to logic_tree.png
    print("Logic tree saved as logic_tree.png")


def create_mapping_flowchart():
    """
    Create a flowchart of the logic-to-text mapping process using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    steps = ["Logic", "Parse", "Select", "Plan", "Text"]
    for i, step in enumerate(steps):
        ax.text(
            i,
            0.5,
            step,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightblue", edgecolor="black"),
        )
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((i + 0.4, 0.5), (i + 0.6, 0.5), mutation_scale=20)
            ax.add_patch(arrow)
    ax.axis("off")
    plt.savefig("mapping_flowchart.png")
    plt.close()
    print("Flowchart saved as mapping_flowchart.png")


def process_logic2text_dataset():
    """
    Process a Logic2Text dataset example to map logic to text.
    Requires: datasets library
    """
    try:
        dataset = load_dataset("logic2text", split="train[:1]")
        example = dataset[0]
        print("Dataset example:", example)

        # Simple rule for population logic
        def map_population(logic, table):
            if "Max(Population)" in logic:
                city = logic.split("=")[1].strip()
                pop = table[0]["Population"]
                return f"{city} has the highest population, {pop}."

        # Test with example
        table = [{"City": "New York", "Population": "8M"}]
        logic = "Max(Population) = NewYork"
        print("Logic2Text output:", map_population(logic, table))
    except Exception as e:
        print(f"Error: {e}. Install datasets (pip install datasets) or check internet.")


# Run all functions
if __name__ == "__main__":
    print("Creating visualizations and processing dataset...")
    create_logic_tree()
    create_mapping_flowchart()
    process_logic2text_dataset()

    # Research Lab: Modify the map_population function to handle more logic types
    # Example idea: Add rules for "Min(Population)" or comparisons (e.g., Greater(Pop1, Pop2)).
