# rouge_implementation.py
# ROUGE-N and ROUGE-L Implementation

from collections import Counter


def rouge_n(candidate, reference, n=1):
    """
    ROUGE-N: n-gram overlap (recall)
    """
    cand_ngrams = Counter(
        [tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1)]
    )
    ref_ngrams = Counter(
        [tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)]
    )

    matches = sum((cand_ngrams & ref_ngrams).values())
    total_ref = sum(ref_ngrams.values())

    return matches / total_ref if total_ref > 0 else 0


def lcs_length(X, Y):
    """
    Longest Common Subsequence using DP
    """
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def rouge_l(candidate, reference):
    """
    ROUGE-L: Longest Common Subsequence
    """
    lcs = lcs_length(candidate, reference)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


# === TEST ===
if __name__ == "__main__":
    cand = "police killed the gunman".split()
    ref = "police kill the gunman".split()

    print(f"ROUGE-1: {rouge_n(cand, ref, 1):.3f}")
    print(f"ROUGE-2: {rouge_n(cand, ref, 2):.3f}")
    print(f"ROUGE-L: {rouge_l(cand, ref):.3f}")
