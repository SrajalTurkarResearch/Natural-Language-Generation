"""
utils.py: Core mathematical utilities for hallucination detection.
Author: Inspired by Turing, Einstein, Tesla – Crafted for eternal research.
"""

import numpy as np
from typing import List, Union


def semantic_entropy(probs: Union[List[float], np.ndarray]) -> float:
    """
    Compute semantic entropy for hallucination detection.

    Theory: Measures uncertainty in semantic clusters. High entropy indicates potential hallucinations.
    Formula: H = -∑ P_i * log2(P_i)

    Args:
        probs: Probability distribution over semantic clusters.

    Returns:
        Entropy value (float). Threshold example: >1.0 flags hallucination.

    Example:
        >>> semantic_entropy([0.6, 0.3, 0.1])
        1.295 (approx)
    """
    probs = np.array(probs)
    probs = probs / np.sum(probs)  # Normalize if needed
    return -np.sum(probs * np.log2(probs + 1e-10))  # Avoid log(0)


def consistency_score(similarities: np.ndarray) -> float:
    """
    Calculate consistency from a similarity matrix.

    Logic: Average off-diagonal similarities from multiple generations.
    Low score suggests inconsistency, hence hallucination.

    Args:
        similarities: Cosine similarity matrix (n x n).

    Returns:
        Mean consistency score (0-1).
    """
    n = similarities.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    return np.mean(similarities[triu_indices])


def tsv_separation(
    mu_t: np.ndarray, mu_h: np.ndarray, sigma_t: float, sigma_h: float
) -> float:
    """
    Compute separation metric for Truthfulness Separator Vector (TSV).

    Advanced Math: |μ_truthful - μ_hallucinated| / (σ_t + σ_h)
    High value indicates good separability in latent space.

    Args:
        mu_t, mu_h: Mean vectors for truthful and hallucinated states.
        sigma_t, sigma_h: Standard deviations.

    Returns:
        Separation score.

    Example:
        >>> tsv_separation(np.array([0.5]), np.array([0.1]), 0.1, 0.2)
        1.333
    """
    diff = np.linalg.norm(mu_t - mu_h)
    return diff / (sigma_t + sigma_h + 1e-10)
