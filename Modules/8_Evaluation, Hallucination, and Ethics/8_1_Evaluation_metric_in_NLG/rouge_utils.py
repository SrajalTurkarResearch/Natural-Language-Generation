from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict


def compute_rouge(
    hypothesis: str, reference: str, metrics: List[str] = ["rouge1", "rougeL"]
) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-L) for a hypothesis against a reference.

    Args:
        hypothesis (str): Machine-generated text.
        reference (str): Human-written reference.
        metrics (List[str]): ROUGE variants to compute (default: ['rouge1', 'rougeL']).

    Returns:
        Dict[str, float]: Dictionary of F1 scores for each metric.

    Raises:
        ValueError: If inputs are empty or metrics are invalid.

    Research Note:
        ROUGE-L is ideal for summarization as it captures structural overlap. Test multiple references to improve robustness.
    """
    if not hypothesis or not reference:
        raise ValueError("Hypothesis and reference cannot be empty.")

    valid_metrics = ["rouge1", "rouge2", "rougeL"]
    if not all(m in valid_metrics for m in metrics):
        raise ValueError(f"Metrics must be in {valid_metrics}")

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {metric: scores[metric].fmeasure for metric in metrics}


def visualize_rouge(
    hypotheses: List[str], references: List[str], metric: str = "rougeL"
) -> None:
    """
    Visualize ROUGE scores across multiple hypothesis-reference pairs.

    Args:
        hypotheses (List[str]): List of machine-generated texts.
        references (List[str]): List of reference texts.
        metric (str): ROUGE variant to visualize (default: 'rougeL').

    Research Note:
        Heatmaps for ROUGE-L can highlight structural matches. Compare with BERTScore for abstractive summaries.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses must match number of references.")

    scores = [
        compute_rouge(hyp, ref, [metric])[metric]
        for hyp, ref in zip(hypotheses, references)
    ]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=range(len(scores)), y=scores, palette="magma")
    plt.title(f"{metric.upper()} Score Comparison")
    plt.xlabel("Sentence Pair Index")
    plt.ylabel(f"{metric.upper()} F1 Score (0-1)")
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    # Example usage
    ref = "The cat sits on the mat."
    hyp = "A feline sat on the rug."
    scores = compute_rouge(hyp, ref)
    print(f"ROUGE Scores: {scores}")

    # Visualize multiple examples
    refs = [ref, "The dog runs in the park."]
    hyps = [hyp, "A puppy jogs in the garden."]
    visualize_rouge(hyps, refs)
