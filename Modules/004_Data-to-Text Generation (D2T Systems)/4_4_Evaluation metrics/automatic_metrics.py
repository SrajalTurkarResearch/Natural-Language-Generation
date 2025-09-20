# automatic_metrics.py
# A beginner-friendly tutorial on NLG evaluation metrics: Automatic Metrics
# Covers word-based (BLEU, ROUGE, METEOR) and embedding-based metrics (BERTScore, MoverScore)
# Includes practical code to compute metrics

"""
Automatic Evaluation Metrics
===========================
Automatic metrics are fast, scalable algorithms to score NLG outputs. They include:

1. Word-Based Metrics
--------------------
BLEU (Bilingual Evaluation Understudy)
- Measures: N-gram overlap with reference text
- Range: 0–1 (higher = better)
- Analogy: Checks if your dish uses the same ingredients as a reference recipe
Example:
- Reference: "The cat is on the mat"
- Generated: "The cat sits on the mat"
- BLEU counts matching n-grams (e.g., "the cat"). High overlap = high score.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- Measures: Recall of n-grams or longest common subsequences
- Variants: ROUGE-N, ROUGE-L
- Analogy: Checks how many reference ingredients you included
Example:
- Reference: "The quick brown fox jumps"
- Generated: "The brown fox jumps quickly"
- ROUGE-L looks for longest matching sequence ("brown fox jumps").

METEOR
- Measures: Matches exact words, synonyms, stems, plus word order
- Range: 0–1
- Analogy: Allows substitute ingredients (e.g., basil for parsley)
Example:
- Reference: "The dog runs fast"
- Generated: "The puppy quickly runs"
- METEOR matches "dog" with "puppy" (synonym).

2. Embedding-Based Metrics
-------------------------
BERTScore
- Measures: Semantic similarity using BERT embeddings
- Range: -1 to 1
- Analogy: Compares flavor profiles of dishes
Example:
- Reference: "The sun sets slowly"
- Generated: "The sun gradually descends"
- High score due to similar meaning.

MoverScore
- Measures: Semantic distance using Word Mover’s Distance
- Analogy: Measures effort to transform one dish’s flavors into another’s.

3. Other Metrics
---------------
Perplexity: Measures fluency (lower = more predictable). Analogy: Grades natural speech.
Distinct-n: Measures diversity (unique n-grams). Analogy: Counts unique spices.

Instructions to Run
==================
1. Install libraries: `pip install nltk rouge_score bert-score`
2. Save as `automatic_metrics.py`
3. Run: `python automatic_metrics.py`
4. Outputs BLEU, ROUGE, and BERTScore for sample texts
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Download NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Sample texts
reference = "The cat is on the mat"
generated = "The cat sits on the mat"

# BLEU Score
ref_tokens = [reference.split()]
gen_tokens = generated.split()
bleu_score = sentence_bleu(ref_tokens, gen_tokens, weights=(0.5, 0.5))
print(f"BLEU Score: {bleu_score:.3f}")

# ROUGE Score
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
rouge_scores = scorer.score(reference, generated)
print(f"ROUGE-1 Recall: {rouge_scores['rouge1'].recall:.3f}")
print(f"ROUGE-L Recall: {rouge_scores['rougeL'].recall:.3f}")

# BERTScore
refs = [reference]
cands = [generated]
P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
print(f"BERTScore F1: {F1.mean().item():.3f}")

if __name__ == "__main__":
    print("NLG Evaluation Tutorial: Automatic Metrics")
    print("Run complete. Check outputs above.")
    print(
        "Next: Run `visualizations_applications.py` for visualizations and applications."
    )
