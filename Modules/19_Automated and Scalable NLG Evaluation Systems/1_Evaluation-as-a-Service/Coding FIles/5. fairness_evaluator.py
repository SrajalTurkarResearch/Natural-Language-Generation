# fairness_evaluator.py
# Fairness Evaluation in NLG Outputs


def demographic_parity_score(texts_male, texts_female, scorer_func):
    """
    Check if NLG treats male/female contexts equally
    scorer_func: function that scores text (e.g., sentiment, formality)
    """
    scores_male = [scorer_func(text) for text in texts_male]
    scores_female = [scorer_func(text) for text in texts_female]

    avg_male = sum(scores_male) / len(scores_male)
    avg_female = sum(scores_female) / len(scores_female)

    parity_gap = abs(avg_male - avg_female)
    return {
        "avg_male": round(avg_male, 3),
        "avg_female": round(avg_female, 3),
        "parity_gap": round(parity_gap, 3),
        "fair": parity_gap < 0.05,
    }


# Example: Sentiment scorer (dummy)
def dummy_sentiment(text):
    positive = ["good", "great", "happy", "excellent"]
    negative = ["bad", "sad", "terrible"]
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    return (pos - neg) / len(words) if words else 0


# === TEST ===
if __name__ == "__main__":
    male_context = ["He is a great leader.", "The boy won the game."]
    female_context = ["She is a good nurse.", "The girl helped her mom."]

    result = demographic_parity_score(male_context, female_context, dummy_sentiment)
    print("Fairness Report:")
    for k, v in result.items():
        print(f"  {k}: {v}")
