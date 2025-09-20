# project_utils.py
# Author: Grok (Tesla's blueprint for innovation)
# Purpose: Utilities for NLG projects on real datasets.
# Usage: from project_utils import load_bias_dataset

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_bias_dataset(dataset_name="AlexaAI/bold"):
    """
    Load a bias dataset for testing.

    Parameters:
    - dataset_name (str): Hugging Face dataset.

    Returns:
    - Dataset: Loaded data.
    """
    return load_dataset(dataset_name, split="test")


def build_safe_generator(model_name="gpt2"):
    """
    Initialize a safe NLG model with tokenizer.

    Returns:
    - tuple: (model, tokenizer)

    Project Idea: Fine-tune on fair data, add guards.
    100-Year Tip: Extend to multimodal (text+image) in future.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer


# Example:
if __name__ == "__main__":
    ds = load_bias_dataset()
    print(ds[0])
