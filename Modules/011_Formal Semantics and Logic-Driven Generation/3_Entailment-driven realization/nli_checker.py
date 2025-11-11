# nli_checker.py
# Purpose: Check if a premise entails a hypothesis using an NLI model
# For aspiring scientists: Learn how NLI ensures text faithfulness
# Dependencies: transformers (pip install transformers)

from transformers import pipeline


def check_entailment(premise, hypothesis, model):
    """
    Check if premise entails hypothesis using NLI model.
    Args:
        premise (str): Starting fact
        hypothesis (str): Text to verify
        model: Hugging Face NLI pipeline
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

    # Test pair
    premise = "Kids play soccer in the park."
    hypothesis = "Children are playing a sport."
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(
        f"Entailment: {check_entailment(premise, hypothesis, nli_model)}"
    )  # Expected: True
