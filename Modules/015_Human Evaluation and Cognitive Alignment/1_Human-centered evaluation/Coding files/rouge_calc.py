# rouge_calc.py: Compute ROUGE Score for NLG Evaluation
# Theory: ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap in summaries.
# Variants: ROUGE-1 (unigrams), ROUGE-L (longest common subsequence).
# Logic: Stem words for flexibility; scores include precision, recall, F1.
# As a mathematician, note: F1 = 2PR / (P + R), balancing precision (P) and recall (R).

from rouge_score import rouge_scorer

# Initialize scorer with desired metrics
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

# Example texts
reference = "Reference text for evaluation."
candidate = "Generated text for eval."

# Compute scores
scores = scorer.score(reference, candidate)
print("ROUGE Scores:", scores)

# Breakdown: For ROUGE-1, if overlap=5 words, ref=10, cand=8: Recall=0.5, Precision=0.625, F1≈0.556
# Research Reflection: Low F1 signals need for human-centered refinements.
