"""
fairness_metrics.py

A module for computing fairness metrics in NLG outputs.
Designed for researchers to ensure equitable model behavior.
Supports Demographic Parity and other metrics.

Dependencies: fairlearn, numpy, scikit-learn
"""

import numpy as np
from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import confusion_matrix


def calculate_demographic_parity(predictions: list, groups: list) -> float:
    """
    Calculate Demographic Parity difference between groups.

    Formula: |P(Y=1|A=0) - P(Y=1|A=1)|

    Args:
        predictions (list): Binary predictions (1=positive, 0=negative).
        groups (list): Group labels (e.g., 0=male, 1=female).

    Returns:
        float: Demographic Parity difference (0 = fair, higher = unfair).

    Example:
        >>> preds = [1, 1, 0, 1, 0]  # Positive/negative outputs
        >>> groups = [0, 0, 1, 1, 1]  # Male/Female
        >>> calculate_demographic_parity(preds, groups)
        0.3333333333333333
    """
    try:
        if len(predictions) != len(groups):
            raise ValueError("Predictions and groups must have same length.")
        return demographic_parity_difference(
            predictions, groups, sensitive_features=groups
        )
    except Exception as e:
        print(f"Error in calculate_demographic_parity: {e}")
        return None


def confusion_matrix_per_group(y_true: list, y_pred: list, groups: list) -> dict:
    """
    Compute confusion matrices for each group.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        groups (list): Group labels.

    Returns:
        dict: Confusion matrices for each group.

    Example:
        >>> y_true = [1, 0, 1, 0, 1]
        >>> y_pred = [1, 1, 0, 0, 1]
        >>> groups = [0, 0, 1, 1, 1]
        >>> confusion_matrix_per_group(y_true, y_pred, groups)
        {'group_0': array([[0, 1], [0, 1]]), 'group_1': array([[1, 0], [1, 1]])}
    """
    try:
        unique_groups = np.unique(groups)
        matrices = {}
        for group in unique_groups:
            mask = [g == group for g in groups]
            y_true_g = [y_true[i] for i in range(len(y_true)) if mask[i]]
            y_pred_g = [y_pred[i] for i in range(len(y_pred)) if mask[i]]
            matrices[f"group_{group}"] = confusion_matrix(y_true_g, y_pred_g)
        return matrices
    except Exception as e:
        print(f"Error in confusion_matrix_per_group: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    preds = [1, 1, 0, 1, 0]
    groups = [0, 0, 1, 1, 1]
    y_true = [1, 0, 1, 0, 1]
    print(f"Demographic Parity: {calculate_demographic_parity(preds, groups):.3f}")
    print(f"Confusion Matrices: {confusion_matrix_per_group(y_true, preds, groups)}")
