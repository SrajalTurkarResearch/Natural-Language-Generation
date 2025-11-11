"""
perplexity_calculator.py
========================
Module 7: Compute Perplexity (NLG Quality Score)
Lower = Better Model
"""

import math


def calculate_perplexity(probabilities: list) -> float:
    """
    Calculate perplexity from word probabilities.

    Args:
        probabilities: List of P(word) for each word in sentence

    Returns:
        Perplexity score
    """
    if not probabilities or any(p <= 0 for p in probabilities):
        return float("inf")

    log_sum = sum(math.log2(p) for p in probabilities)
    avg_log_prob = log_sum / len(probabilities)
    perplexity = 2 ** (-avg_log_prob)

    return perplexity


# === EXAMPLE ===
if __name__ == "__main__":
    print("📊 PERPLEXITY CALCULATOR\n")

    # Good model
    good_probs = [0.9, 0.85, 0.8, 0.95]
    ppl_good = calculate_perplexity(good_probs)

    # Bad model
    bad_probs = [0.3, 0.2, 0.1, 0.4]
    ppl_bad = calculate_perplexity(bad_probs)

    print(f"Sentence: 'I am very happy'")
    print(f" Good Model PPL: {ppl_good:.3f} ← Better!")
    print(f"  Bad Model PPL: {ppl_bad:.3f}\n")

    print("🧠 Insight: Human-level PPL ≈ 1.0–1.5")
