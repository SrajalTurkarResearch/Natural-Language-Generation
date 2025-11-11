# project_journalism_summary.py
# Real-World: Automated Sports News from Game Stats
# Use: ESPN, BBC, Automated Journalism

from utils import neural_summary_nlg


def sports_news_nlg(stats):
    """
    Input: Game stats dict
    Output: News-style summary
    """
    team_a = stats["team_a"]
    team_b = stats["team_b"]
    score_a = stats["score_a"]
    score_b = stats["score_b"]

    # === SYMBOLIC: Determine winner & margin ===
    if score_a > score_b:
        winner = team_a
        loser = team_b
        margin = score_a - score_b
        result = f"{team_a} defeated {team_b}"
    else:
        winner = team_b
        loser = team_a
        margin = score_b - score_a
        result = f"{team_b} defeated {team_a}"

    if margin == 1:
        intensity = "in a nail-biter"
    elif margin <= 3:
        intensity = "in a close contest"
    else:
        intensity = "convincingly"

    # === NEURAL: Fluent headline & body ===
    body = neural_summary_nlg(
        f"{team_a} scored {score_a}, {team_b} scored {score_b}. "
        f"The game was competitive with strong defense."
    )

    headline = f"{winner} Defeats {loser} {score_a}-{score_b} {intensity.title()}!"

    return f"HEADLINE: {headline}\n\n{body}"


# === TEST ===
if __name__ == "__main__":
    game = {"team_a": "Lakers", "team_b": "Warriors", "score_a": 105, "score_b": 108}
    print(sports_news_nlg(game))
