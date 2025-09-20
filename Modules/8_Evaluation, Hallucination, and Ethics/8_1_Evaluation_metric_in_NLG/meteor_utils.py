from meteor import meteor_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List


def compute_meteor(hypothesis: str, reference: str) -> float:
    """
    Compute METEOR score for a hypothesis against a reference.

    Args:
        hypothesis (str): Machine-generated text.
        reference (str): Human-written reference.

    Returns:
        float: METEOR score (0-1).

    Raises:
        ValueError: If inputs are empty.

    Research Note:
        METEOR’s synonym matching makes it suitable for translation tasks. Investigate its performance in low-resource languages.
    """
    if not hypothesis or not reference:
        raise ValueError("Hypothesis and reference cannot be empty.")

    score = meteor_score([reference], hypothesis)
    return score


def visualize_meteor(
    hypotheses: List[str], references: List[str], title: str = "METEOR Score Comparison"
) -> None:
    """
    Visualize METEOR scores across multiple hypothesis-reference pairs.

    Args:
        hypotheses (List[str]): List of machine-generated texts.
        references (List[str]): List of reference texts.
        title (str): Plot title.

    Research Note:
        METEOR’s alignment visualization can reveal synonym matches. Compare with BLEU to highlight differences.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses must match number of references.")

    scores = [compute_meteor(hyp, ref) for hyp, ref in zip(hypotheses, references)]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=range(len(scores)), y=scores, palette="coolwarm")
    plt.title(title)
    plt.xlabel("Sentence Pair Index")
    plt.ylabel("METEOR Score (0-1)")
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    # Example usage
    ref = "The cat sits on the mat."
    hyp = "A feline sat on the rug."
    score = compute_meteor(hyp, ref)
    print(f"METEOR Score: {score:.3f}")

    # Visualize multiple examples
    refs = [ref, "The dog runs in the park."]
    hyps = [hyp, "A puppy jogs in the garden."]
    visualize_meteor(hyps, refs)
