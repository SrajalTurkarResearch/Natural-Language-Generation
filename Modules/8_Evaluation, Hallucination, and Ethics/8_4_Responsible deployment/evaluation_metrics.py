# evaluation_metrics.py
# Author: Grok (Einstein's equations for AI measurement)
# Purpose: Compute NLG metrics responsibly.
# Usage: from evaluation_metrics import compute_perplexity

import math
from detoxify import Detoxify  # Requires pip install detoxify


def compute_perplexity(probs):
    """
    Calculate perplexity for generation quality.

    Parameters:
    - probs (list): Word probabilities.

    Returns:
    - float: Perplexity (lower=better).

    Math: 2^(-1/N * sum log P(w_i))
    """
    N = len(probs)
    if N == 0:
        return float("inf")
    log_sum = sum(math.log(p) for p in probs if p > 0)
    return math.pow(2, -(1 / N) * log_sum)


def check_toxicity(text, threshold=0.5):
    """
    Detect toxicity in generated text.

    Parameters:
    - text (str): NLG output.
    - threshold (float): Block if above.

    Returns:
    - bool: True if toxic.
    """
    detector = Detoxify("original")
    score = detector.predict(text)["toxicity"]
    return score > threshold


# Example:
if __name__ == "__main__":
    print("Perplexity:", compute_perplexity([0.9, 0.8, 0.9]))
