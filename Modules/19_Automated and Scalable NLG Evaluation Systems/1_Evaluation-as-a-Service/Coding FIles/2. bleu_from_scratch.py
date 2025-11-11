# bleu_from_scratch.py
# BLEU Score Implementation from Scratch
# No external libraries needed except math and collections

from collections import Counter
import math


def ngrams(tokens, n):
    """Generate n-grams from token list"""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def modified_precision(candidate, reference, n):
    """
    Compute modified n-gram precision
    """
    cand_ngrams = Counter(ngrams(candidate, n))
    ref_ngrams = Counter(ngrams(reference, n))

    matches = 0
    for gram in cand_ngrams:
        matches += min(cand_ngrams[gram], ref_ngrams.get(gram, 0))

    total = sum(cand_ngrams.values())
    return matches / total if total > 0 else 0


def brevity_penalty(candidate, reference):
    """
    Penalize short candidates
    """
    c_len = len(candidate)
    r_len = len(reference)
    if c_len > r_len:
        return 1.0
    elif c_len == 0:
        return 0.0
    else:
        return math.exp(1 - r_len / c_len)


def bleu_score(candidate, reference, max_n=4):
    """
    Full BLEU score with n=1 to 4
    """
    candidate = candidate.lower().split()
    reference = reference.lower().split()

    if len(candidate) == 0:
        return 0.0

    # Compute precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        p = modified_precision(candidate, reference, n)
        precisions.append(p)

    # Geometric mean
    log_sum = sum(math.log(p) for p in precisions if p > 0)
    geo_mean = math.exp(log_sum / max_n)

    # Final BLEU
    bp = brevity_penalty(candidate, reference)
    return bp * geo_mean


# === TEST ===
if __name__ == "__main__":
    cand = "The cat is on the mat"
    ref = "There is a cat on the mat"

    score = bleu_score(cand, ref)
    print(f"Generated: {cand}")
    print(f"Reference: {ref}")
    print(f"BLEU Score: {score:.4f}")
