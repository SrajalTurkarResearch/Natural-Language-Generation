# helpfulness_eval.py
# --------------------------------------------------------------
# Helpfulness evaluation – BLEU, ROUGE, and a tiny G-Eval stub
# --------------------------------------------------------------

# 1. Install (first run only)
# --------------------------------------------------------------
# !pip install nltk rouge-score

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download("punkt")

# --------------------------------------------------------------
# 2. BLEU score
# --------------------------------------------------------------
reference = [["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]]
candidate = ["Quick", "brown", "fox", "jumps", "over", "lazy", "dog"]

smooth = SmoothingFunction().method1
bleu = sentence_bleu(reference, candidate, smoothing_function=smooth)
print(f"BLEU = {bleu:.4f}")

# --------------------------------------------------------------
# 3. ROUGE score
# --------------------------------------------------------------
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
ref_str = "The quick brown fox jumps over the lazy dog."
cand_str = "Quick brown fox jumps over lazy dog."
scores = scorer.score(ref_str, cand_str)
print(f"ROUGE-L F1 = {scores['rougeL'].fmeasure:.4f}")

# --------------------------------------------------------------
# 4. Tiny G-Eval (LLM-as-judge) – just a prompt example
# --------------------------------------------------------------
from transformers import pipeline

judge = pipeline("text-generation", model="gpt2")
prompt = f"""Rate the helpfulness of the following answer on a 1-5 scale.
Question: How do I change a flat tire?
Answer: Use a jack, remove the nuts, replace the tire, tighten nuts.
Rating:"""

rating_text = judge(prompt, max_length=60, num_return_sequences=1)[0]["generated_text"]
print("\n=== G-Eval (demo) ===\n", rating_text)
