"""
data_handlers.py: Dataset loading and processing for hallucination research.
Supports Hugging Face datasets for reproducibility.
"""

from datasets import load_dataset
import pandas as pd
from typing import Optional


def load_hallucination_dataset(
    name: str = "truthful_qa", split: str = "validation"
) -> pd.DataFrame:
    """
    Load a standard dataset like TruthfulQA or HaluEval.

    Args:
        name: Dataset name (e.g., 'truthful_qa').
        split: Train/validation/test.

    Returns:
        Pandas DataFrame for easy manipulation.

    Example:
        >>> df = load_hallucination_dataset()
        >>> print(df.head())
    """
    dataset = load_dataset(name, split=split)
    return pd.DataFrame(dataset)


def preprocess_responses(responses: list, labels: Optional[list] = None) -> dict:
    """
    Preprocess raw responses for detection.

    Logic: Clean text, optionally align with labels for supervised metrics.
    """
    cleaned = [resp.strip() for resp in responses]
    if labels:
        return {"responses": cleaned, "labels": labels}
    return {"responses": cleaned}


# Advanced: Custom dataset builder (for your 100-year archive)
def create_custom_dataset(queries: list, ground_truths: list) -> pd.DataFrame:
    """
    Build a DataFrame for personal experiments.
    """
    return pd.DataFrame({"query": queries, "ground_truth": ground_truths})
