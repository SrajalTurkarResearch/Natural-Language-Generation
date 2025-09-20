"""
visualizations.py: Plotting functions for hallucination analysis.
Use matplotlib for timeless, publication-quality visuals.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def plot_entropy_distribution(
    entropies: List[float], threshold: float = 1.0, save_path: Optional[str] = None
):
    """
    Histogram of entropy values.

    Insight: Visualize uncertainty distribution across queries.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(entropies, bins=20, color="skyblue", edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="--", label="Hallucination Threshold")
    plt.title("Semantic Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_latent_separation(
    truthful: np.ndarray, hallucinated: np.ndarray, save_path: Optional[str] = None
):
    """
    Scatter plot for TSV latent space.

    Args:
        truthful, hallucinated: 2D arrays of points (samples x 2 dims).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(truthful[:, 0], truthful[:, 1], label="Truthful", color="green")
    plt.scatter(
        hallucinated[:, 0], hallucinated[:, 1], label="Hallucinated", color="red"
    )
    plt.title("Latent Space Separation (TSV)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_consistency_matrix(sim_matrix: np.ndarray, save_path: Optional[str] = None):
    """
    Heatmap of similarity matrix from self-consistency.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis")
    plt.colorbar()
    plt.title("Response Similarity Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()
