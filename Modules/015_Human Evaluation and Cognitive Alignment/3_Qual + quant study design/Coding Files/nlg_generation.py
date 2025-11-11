# nlg_generation.py: Basic NLG text generation example.
# This script demonstrates how to use a pre-trained model for generating text,
# a fundamental step in NLG research for creating human-like outputs from prompts.
# As a scientist, start here to understand generation before evaluation.

from transformers import pipeline
import nltk

nltk.download("punkt")  # Download tokenizer data if needed.

# Load a text generation pipeline using GPT-2 (a simple transformer model).
generator = pipeline("text-generation", model="gpt2")

# Generate text from a prompt. max_length controls output size.
prompt = "The future of AI is"
generated_text = generator(prompt, max_length=50)[0]["generated_text"]

# Output the result for inspection.
print("Generated Text:")
print(generated_text)

# Research Tip: In qual studies, analyze if this feels 'natural'; in quant, measure coherence.
