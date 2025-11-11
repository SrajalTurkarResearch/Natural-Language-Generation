# visualize_metrics.py
"""
Purpose: Visualize execution-based evaluation metrics (EX, F1) for NLG queries.
Creates bar plots to compare performance across queries. Designed for aspiring
scientists to understand NLG evaluation intuitively.

Dependencies: matplotlib, seaborn, pandas

Usage: Run to generate a bar plot of sample metrics. Extend for research by adding
more metrics or visualizing real dataset results (e.g., Spider).

Author: Inspired by Einstein’s visual thought experiments and Feynman’s diagrams.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_metrics(queries, ex_scores, f1_scores):
    """
    Plot Execution Accuracy and F1 Scores as a bar plot.
    Args:
        queries (list): Query names (e.g., ['Q1', 'Q2']).
        ex_scores (list): Execution Accuracy scores (%).
        f1_scores (list): F1 scores.
    """
    # Create DataFrame for plotting
    data = pd.DataFrame(
        {
            "Query": queries * 2,
            "Score": ex_scores + [f * 100 for f in f1_scores],  # Scale F1 to %
            "Metric": ["EX"] * len(queries) + ["F1 (scaled)"] * len(queries),
        }
    )

    # Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Query", y="Score", hue="Metric", data=data)
    plt.ylabel("Score (%)")
    plt.title("Execution-Based Evaluation Metrics")
    plt.legend()
    plt.show()


def main():
    # Sample data (from evaluation results)
    queries = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    ex_scores = [70, 80, 60, 90, 85]  # Execution Accuracy (%)
    f1_scores = [0.65, 0.75, 0.55, 0.88, 0.82]  # F1 Scores

    print("Generating visualization...")
    plot_metrics(queries, ex_scores, f1_scores)


if __name__ == "__main__":
    main()

# Scientist Tip: Modify to plot real results from evaluate_nlg.py or a dataset like
# Spider. Research idea: Visualize metric trade-offs (e.g., EX vs. efficiency).
