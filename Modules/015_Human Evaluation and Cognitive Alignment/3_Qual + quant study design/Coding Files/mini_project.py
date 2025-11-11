# mini_project.py: Mini project for NLG generation and basic evaluation.
# This combines generation with quant eval, a hands-on exercise for beginners.
# Simulate a small study: Generate text, then score against a reference.

from transformers import pipeline
import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")

# Step 1: Generate text.
generator = pipeline("text-generation", model="gpt2")
prompt = "Explain NLG"
gen_text = generator(prompt, max_length=100)[0]["generated_text"]
print("Generated Text:")
print(gen_text)

# Step 2: Prepare tokenized reference and candidate.
reference_text = prompt + " is the process of generating text from data."
ref_tokens = [nltk.word_tokenize(reference_text)]  # List of lists.
cand_tokens = nltk.word_tokenize(gen_text)

# Step 3: Compute BLEU.
score = sentence_bleu(ref_tokens, cand_tokens)
print(f"BLEU Score for Mini Project: {score:.4f}")

# Project Extension: Add qual by manually noting 'does it explain well?' – mix methods!
