# snli_entailment_checker.py
# Purpose: Mini project to check entailment on SNLI dataset
# For aspiring scientists: Practice with real NLP datasets
# Dependencies: transformers, datasets (pip install transformers datasets)

from transformers import pipeline
from datasets import load_dataset


def check_entailment(premise, hypothesis, model):
    """
    Check if premise entails hypothesis.
    Args:
        premise (str): Starting fact
        hypothesis (str): Text to verify
        model: NLI pipeline
    Returns:
        bool: True if entailment score > 0.7
    """
    input_text = f"{premise} [SEP] {hypothesis}"
    result = model(input_text)
    return result[0]["label"] == "entailment" and result[0]["score"] > 0.7


# Example usage
if __name__ == "__main__":
    # Load NLI model
    nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

    # Load SNLI dataset (small subset)
    snli = load_dataset("snli", split="test[:100]")

    # Test first pair
    premise = snli[0]["premise"]
    hypothesis = snli[0]["hypothesis"]
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Entailment: {check_entailment(premise, hypothesis, nli_model)}")
