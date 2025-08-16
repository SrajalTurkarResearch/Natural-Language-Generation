# nlg_visualizations.py
# Visualizations for dataset sizes and hypothetical model performance
# Uses Matplotlib and Seaborn for clear, beginner-friendly plots

import matplotlib.pyplot as plt
import seaborn as sns


def plot_dataset_sizes():
    """
    Plots a bar chart comparing the sizes of WebNLG, DART, and ToTTo.
    """
    print("\n## Visualizing Dataset Sizes\n")

    datasets = ["WebNLG", "DART", "ToTTo"]
    sizes = [35000, 82191, 120000]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=datasets, y=sizes, palette="viridis")
    plt.title("Dataset Size Comparison")
    plt.ylabel("Number of Examples")
    plt.xlabel("Dataset")
    plt.show()
    print("Plot shows the number of examples in each dataset.")


def plot_model_performance():
    """
    Plots hypothetical BLEU scores for a model on WebNLG, DART, and ToTTo.
    """
    print("\n## Visualizing Hypothetical Model Performance\n")

    datasets = ["WebNLG", "DART", "ToTTo"]
    bleu_scores = [65, 60, 70]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=datasets, y=bleu_scores, palette="magma")
    plt.title("Hypothetical BLEU Scores Across Datasets")
    plt.ylabel("BLEU Score (%)")
    plt.xlabel("Dataset")
    plt.show()
    print("Plot shows hypothetical BLEU scores for a model.")


if __name__ == "__main__":
    plot_dataset_sizes()
    plot_model_performance()
