# bleu_score_calc.py: Quantitative evaluation using BLEU score.
# BLEU measures n-gram overlap between generated and reference text,
# a key metric in NLG for assessing accuracy and fluency.
# Logic: Higher score means better match; useful in deductive quant research.

import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")  # Ensure tokenizer is available.

# Tokenized reference text (list of lists for multiple refs, but here one).
reference = [["This", "is", "a", "test", "sentence"]]

# Candidate (generated) text tokens.
candidate = ["This", "is", "test", "sentence"]

# Compute BLEU score (default: up to 4-grams, with smoothing).
score = sentence_bleu(reference, candidate)

# Output the score.
print(f"BLEU Score: {score:.4f}")

# Scientific Insight: In mixed methods, correlate this with qual themes like 'missing words'.
