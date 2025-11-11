# neurosymbolic_nlg.py
# Neurosymbolic NLG: Rules for structure, neural for fluency.
# Author: Merging symbolic logic (Turing) with neural learning (Bengio).
# Requirements: transformers, torch.
# Usage: Run with sample data.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def rule_based_nlg(data):
    """Symbolic base: Accurate template."""
    return f"The temperature in {data['city']} is {data['temp']}°C."


def neurosymbolic_nlg(data):
    """
    Hybrid generation: Rule base + neural refinement.
    Args:
        data (dict): Input data.
    Returns:
        str: Combined output.
    """
    # Symbolic step: Create constrained base.
    base_text = rule_based_nlg(data)

    # Neural step: Add fluency.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer(base_text + " Additional info:", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0])


if __name__ == "__main__":
    # Test like a scientific trial.
    sample_data = {"city": "Paris", "temp": 25}
    print("Neurosymbolic Output:", neurosymbolic_nlg(sample_data))
    # Insight: Rules prevent errors; neural adds natural flow. Experiment with constraints.
