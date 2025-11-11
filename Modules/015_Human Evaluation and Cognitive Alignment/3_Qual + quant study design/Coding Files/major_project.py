# major_project.py: Major project using a real NLG dataset.
# Loads E2E NLG dataset, prints a sample—extend to generate/evaluate.
# Ideal for research: Quant metrics on dataset + qual analysis of outputs.
# Note: Requires internet for dataset download via Hugging Face.

from datasets import load_dataset
import nltk

nltk.download("punkt")

# Load the E2E NLG dataset (data-to-text generation).
dataset = load_dataset("e2e_nlg")

# Print a sample from the training set.
sample = dataset["train"][0]
print("Sample from E2E NLG Dataset:")
print(f"Input (Meaning Representation): {sample['meaning_representation']}")
print(f"Output (Human Reference): {sample['human_reference']}")

# Next Steps for Your Study:
# - Use a model to generate from input.
# - Compute quant metrics (e.g., BLEU) on generated vs. reference.
# - Qual: Analyze themes in mismatches (e.g., 'lacks detail').
# Scientific Tip: Publish findings like Turing's papers—rigorous and innovative.
