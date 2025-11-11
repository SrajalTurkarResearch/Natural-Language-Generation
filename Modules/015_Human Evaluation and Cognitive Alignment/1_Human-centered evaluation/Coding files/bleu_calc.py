# bleu_calc.py: Compute BLEU Score for NLG Evaluation
# Theory: BLEU (Bilingual Evaluation Understudy) measures how closely generated text matches references via n-gram precision.
# Formula: BLEU = BP * exp(Σ log(p_n) / N), where p_n is n-gram precision, BP is brevity penalty.
# Logic: Tokenize texts, compute overlaps, apply penalty if generated is shorter.
# As Einstein might say, simplicity in metrics reveals truths, but human nuance adds depth.

import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download required resources (run once)
nltk.download("punkt")

# Example data: Reference (gold standard) and candidate (generated text)
reference = [["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]]
candidate = ["Quick", "fox", "jumps", "over", "lazy", "dog"]

# Compute BLEU: Weights default to [0.25]*4 for 1-4 grams
score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {score}")

# Step-by-step calculation example (simplified unigram for illustration)
# Unigram precision: Matching words / Candidate words = 6/6 = 1
# Brevity penalty: exp(1 - ref_len/cand_len) = exp(1 - 9/6) ≈ exp(-0.5) ≈ 0.606
# Simplified BLEU ≈ 1 * 0.606 = 0.606 (actual uses geometric mean for n-grams)
