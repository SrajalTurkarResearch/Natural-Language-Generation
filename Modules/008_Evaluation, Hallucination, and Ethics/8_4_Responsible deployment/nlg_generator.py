# nlg_generator.py
# Author: Grok (inspired by Turing's computational models)
# Purpose: Modular NLG generation with ethical wrappers.
# Usage: from nlg_generator import generate_text

import torch
from transformers import pipeline


def generate_text(prompt, model_name="gpt2", max_length=50, num_sequences=1):
    """
    Generate human-like text using a pre-trained NLG model.

    Parameters:
    - prompt (str): Input text to start generation.
    - model_name (str): Hugging Face model (default: 'gpt2').
    - max_length (int): Max output length.
    - num_sequences (int): Number of outputs.

    Returns:
    - list: Generated texts.

    Ethical Note: Add filters for harm; e.g., check for toxicity post-generation.
    Math Insight: Based on probabilistic sampling; perplexity can be computed separately.
    """
    try:
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )
        outputs = generator(
            prompt, max_length=max_length, num_return_sequences=num_sequences
        )
        return [out["generated_text"] for out in outputs]
    except Exception as e:
        raise ValueError(f"Generation failed: {e}. Ensure transformers is installed.")


# Example usage (for testing):
if __name__ == "__main__":
    print(generate_text("The future of responsible AI is"))
