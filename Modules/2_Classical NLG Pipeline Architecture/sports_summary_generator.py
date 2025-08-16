# sports_summary_generator.py
# Major Project: Generate sports summaries using the Classical NLG Pipeline
# Includes evaluation and visualization

# Install required libraries: pip install nltk matplotlib
import nltk
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Sample game data
game_data = {
    "team_a": "Lions",
    "team_b": "Tigers",
    "score_a": 3,
    "score_b": 2,
    "key_player": "Alex",
    "player_goals": 2,
    "key_moment": "90th minute goal",
}


# Full NLG Pipeline for Sports Summary
def sports_summary_generator(data):
    """Generate a sports summary from game data."""
    # Stage 1: Content Determination
    selected = {
        "team_a": data["team_a"],
        "team_b": data["team_b"],
        "score": f"{data['score_a']}-{data['score_b']}",
        "key_player": data["key_player"],
        "player_goals": data["player_goals"],
        "key_moment": data["key_moment"],
    }
    print("Selected Data:", selected)

    # Stage 2: Document Planning
    plan = {
        "introduction": "Game result",
        "body": {
            "score": selected["score"],
            "player": f"{selected['key_player']} scored {selected['player_goals']} goals",
        },
        "conclusion": selected["key_moment"],
    }
    print("Document Plan:", plan)

    # Stage 3: Microplanning
    sentence = f"{selected['team_a']} defeated {selected['team_b']} {selected['score']}, with {selected['key_player']} scoring {selected['player_goals']} goals."
    conclusion = f"The game was decided by a {selected['key_moment']}."
    microplanned = {"sentence": sentence, "conclusion": conclusion}
    print("Microplanned Content:", microplanned)

    # Stage 4: Surface Realization
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    has_verb = any(tag.startswith("VB") for word, tag in tagged)
    realized = sentence if has_verb else "Error: Sentence lacks a verb."
    print("Realized Sentence:", realized)

    # Stage 5: Post-Processing
    polished = realized + " " + conclusion
    print("Final Text:", polished)

    # Stage 6: Output
    print("Generated Sports Summary:")
    print(polished)
    return polished


# Evaluation with BLEU Score
def evaluate_summary(summary):
    """Calculate BLEU score for the generated summary."""
    reference = [
        "Lions beat Tigers 3-2, with Alex scoring twice. The game ended with a 90th-minute goal.".split()
    ]
    candidate = summary.split()
    bleu = sentence_bleu([reference], candidate)
    print("BLEU Score:", bleu)
    return bleu


# Visualization
def plot_scores(data):
    """Visualize team scores as a bar chart."""
    plt.bar(
        [data["team_a"], data["team_b"]],
        [data["score_a"], data["score_b"]],
        color=["blue", "orange"],
    )
    plt.title("Game Scores")
    plt.ylabel("Goals")
    plt.show()


# Run the major project
if __name__ == "__main__":
    summary = sports_summary_generator(game_data)
    evaluate_summary(summary)
    plot_scores(game_data)
    print("Major Project Completed!")
