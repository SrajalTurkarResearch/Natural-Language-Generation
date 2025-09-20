import sacrebleu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Union


def compute_bleu(
    hypothesis: str, reference: Union[str, List[str]], n_grams: int = 4
) -> float:
    """
    Compute BLEU score for a hypothesis against one or more references.

    Args:
        hypothesis (str): Machine-generated text.
        reference (str or List[str]): Human-written reference(s).
        n_grams (int): Maximum n-gram order (default: 4).

    Returns:
        float: BLEU score (0-100).

    Raises:
        ValueError: If inputs are empty or invalid.

    Research Note:
        BLEU emphasizes precision, making it ideal for translation tasks where exact matches matter.
        Experiment with n-gram weights to improve correlation with human judgments.
    """
    if not hypothesis or not reference:
        raise ValueError("Hypothesis and reference cannot be empty.")

    if isinstance(reference, str):
        reference = [reference]

    bleu = sacrebleu.corpus_bleu([hypothesis], [reference], smooth_method="floor")
    return bleu.score


def visualize_bleu(
    hypotheses: List[str], references: List[str], title: str = "BLEU Score Comparison"
) -> None:
    """
    Visualize BLEU scores across multiple hypothesis-reference pairs.

    Args:
        hypotheses (List[str]): List of machine-generated texts.
        references (List[str]): List of reference texts.
        title (str): Plot title.

    Research Note:
        Bar plots reveal how BLEU penalizes non-exact matches. Consider comparing with BERTScore for semantic tasks.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses must match number of references.")

    scores = [compute_bleu(hyp, ref) for hyp, ref in zip(hypotheses, references)]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=range(len(scores)), y=scores, palette="viridis")
    plt.title(title)
    plt.xlabel("Sentence Pair Index")
    plt.ylabel("BLEU Score (0-100)")
    plt.ylim(0, 100)
    plt.show()


if __name__ == "__main__":
    # Example usage
    ref = "The cat sits on the mat."
    hyp = "A feline sat on the rug."
    score = compute_bleu(hyp, ref)
    print(f"BLEU Score: {score:.3f}")

    # Visualize multiple examples
    refs = [ref, "The dog runs in the park."]
    hyps = [hyp, "A puppy jogs in the garden."]
    visualize_bleu(hyps, refs)
