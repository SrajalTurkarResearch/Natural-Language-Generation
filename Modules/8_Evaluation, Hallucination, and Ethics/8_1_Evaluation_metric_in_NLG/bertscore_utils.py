from bert_score import score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple


def compute_bertscore(
    hypotheses: List[str], references: List[str], lang: str = "en"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute BERTScore (Precision, Recall, F1) for hypotheses against references.

    Args:
        hypotheses (List[str]): Machine-generated texts.
        references (List[str]): Human-written references.
        lang (str): Language code (default: 'en').

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Precision, Recall, F1 scores.

    Raises:
        ValueError: If inputs are empty or lengths mismatch.

    Research Note:
        BERTScore excels in semantic tasks like dialogue. Experiment with newer models (e.g., RoBERTa) for better performance.
    """
    if not hypotheses or not references or len(hypotheses) != len(references):
        raise ValueError(
            "Hypotheses and references must be non-empty and equal in length."
        )

    P, R, F1 = score(hypotheses, references, lang=lang, verbose=False)
    return P.numpy(), R.numpy(), F1.numpy()


def visualize_bertscore(
    hypotheses: List[str], references: List[str], title: str = "BERTScore Comparison"
) -> None:
    """
    Visualize BERTScore F1 scores across multiple hypothesis-reference pairs.

    Args:
        hypotheses (List[str]): Machine-generated texts.
        references (List[str]): Human-written references.
        title (str): Plot title.

    Research Note:
        Heatmaps of token similarities reveal semantic matches. Compare with ROUGE for summarization tasks.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses must match number of references.")

    _, _, f1_scores = compute_bertscore(hypotheses, references)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=range(len(f1_scores)), y=f1_scores, palette="plasma")
    plt.title(title)
    plt.xlabel("Sentence Pair Index")
    plt.ylabel("BERTScore F1 (0-1)")
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    # Example usage
    ref = ["The cat sits on the mat."]
    hyp = ["A feline sat on the rug."]
    P, R, F1 = compute_bertscore(hyp, ref)
    print(f"BERTScore - P: {P[0]:.3f}, R: {R[0]:.3f}, F1: {F1[0]:.3f}")

    # Visualize multiple examples
    refs = [ref[0], "The dog runs in the park."]
    hyps = [hyp[0], "A puppy jogs in the garden."]
    visualize_bertscore(hyps, refs)
