# privacy_utils.py
# Author: Grok (Turing's code-breaking ethics reversed)
# Purpose: Add privacy to NLG data handling.
# Usage: from privacy_utils import add_laplace_noise

import numpy as np


def add_laplace_noise(data, epsilon=0.1, sensitivity=1.0):
    """
    Apply differential privacy via Laplace noise.

    Parameters:
    - data (np.array): Original data (e.g., counts).
    - epsilon (float): Privacy budget (lower=more privacy).
    - sensitivity (float): Max change from one data point.

    Returns:
    - np.array: Noisy data.

    Math: Noise ~ Laplace(0, sensitivity / epsilon).
    Insight: Use in training to prevent memorization (2025 rare: Quantum DP extensions possible).
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise


# Example:
if __name__ == "__main__":
    data = np.array([50, 100])
    print("Noisy:", add_laplace_noise(data))
