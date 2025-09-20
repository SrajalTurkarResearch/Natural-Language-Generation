# Major Project: Sports Summary Generator
#
# Theory:
# - This project generates sports match summaries by aggregating scores and player stats and lexicalizing with engaging language.
# - Aggregation: Combines team scores, player performance, and outcomes.
# - Lexicalization: Uses dynamic verbs (e.g., "defeated") and specific names.
# - Application: Useful for sports journalism or fan apps.
# - Analogy: Like writing a match recap that excites fans by summarizing key moments.
#
# Requirements: Install nltk (`pip install nltk`).
# For your research: Experiment with neural NLG models (e.g., transformers) for richer summaries.

import nltk

nltk.download("punkt")


def sports_summary(data):
    """
    Generates a sports match summary with aggregation and lexicalization.

    Args:
        data (dict): Dictionary with 'TeamA', 'TeamB', 'ScoreA', 'ScoreB', 'TopScorer'.
    Returns:
        str: Summary sentence.
    """
    verb = "defeated" if data["ScoreA"] > data["ScoreB"] else "lost to"
    return f"{data['TeamA']} {verb} {data['TeamB']} {data['ScoreA']}-{data['ScoreB']}, led by {data['TopScorer']}."


# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        "TeamA": "Lakers",
        "TeamB": "Celtics",
        "ScoreA": 110,
        "ScoreB": 105,
        "TopScorer": "LeBron",
    }
    print(sports_summary(data))
    # Output: Lakers defeated Celtics 110-105, led by LeBron.

    # Research Tip: Add more complex aggregation (e.g., include assist stats or game highlights).
