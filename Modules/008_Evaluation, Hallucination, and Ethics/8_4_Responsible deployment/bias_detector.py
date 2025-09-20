# bias_detector.py
# Author: Grok (channeling Einstein's relativity in fairness)
# Purpose: Detect and mitigate biases in NLG outputs.
# Usage: from bias_detector import calculate_demographic_parity

import numpy as np
from fairlearn.metrics import demographic_parity_difference


def calculate_demographic_parity(y_pred, sensitive_features):
    """
    Compute demographic parity for fairness assessment.

    Parameters:
    - y_pred (np.array): Predictions (e.g., 1=positive output).
    - sensitive_features (np.array): Group labels (e.g., 0=male, 1=female).

    Returns:
    - float: Parity difference (closer to 0 = fairer).

    Math: |P(positive|group A) - P(positive|group B)|.
    Example: If high, retrain with balanced data.
    Ethical Insight: From 2025 EU AI Act—audit for high-risk apps.
    """
    # Dummy y_true for metric (all 1s as we're measuring disparity in preds)
    y_true = np.ones_like(y_pred)
    parity = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    return parity


def simple_debias(prompt):
    """
    Basic debiasing via gender swap for testing.

    Parameters:
    - prompt (str): Original prompt.

    Returns:
    - tuple: (original_prompt, debiased_prompt)
    """
    debiased = prompt.replace("He", "They").replace(
        "She", "They"
    )  # Simple neutral swap
    return prompt, debiased


# Example:
if __name__ == "__main__":
    y_pred = np.array([1, 1, 0, 1])
    groups = np.array([0, 0, 1, 1])
    print("Parity:", calculate_demographic_parity(y_pred, groups))
