# rouge_exercise.py: Exercise solution for computing ROUGE score.
# ROUGE measures recall-oriented overlap, complementary to BLEU.
# Use in quant NLG eval for summarization-like tasks.

from rouge_score import rouge_scorer

# Initialize scorer for ROUGE-1 (unigram overlap).
scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

# Sample reference and candidate texts.
reference = "The quick brown fox jumps over the lazy dog"
candidate = "The quick brown dog jumps on the log."

# Compute ROUGE scores.
scores = scorer.score(reference, candidate)

# Output the results.
print("ROUGE-1 Scores:")
print(f"Precision: {scores['rouge1'].precision:.4f}")
print(f"Recall: {scores['rouge1'].recall:.4f}")
print(f"F-Measure: {scores['rouge1'].fmeasure:.4f}")

# Exercise Reflection: As a mathematician, derive why recall matters in NLG—extend to mixed methods by linking to qual feedback.
