# news_article_generator.py
# Real-World NLG: Automated Sports News from Stats
# Used by: Associated Press, BBC, Reuters

import random
from datetime import datetime


def generate_sports_report(match_data):
    """
    Generate full news article from match stats
    Input: dict with team names, scores, key events
    """
    home = match_data["home_team"]
    away = match_data["away_team"]
    home_score = match_data["home_score"]
    away_score = match_data["away_score"]
    events = match_data["events"]  # list of (minute, player, event)

    # Content Planning
    if home_score > away_score:
        winner, loser = home, away
        win_score, lose_score = home_score, away_score
        lead = f"{home} defeated {away} {home_score}–{lose_score}"
    else:
        winner, loser = away, home
        win_score, lose_score = away_score, home_score
        lead = f"{away} beat {home} {away_score}–{home_score}"

    # Key moments
    goals = [e for e in events if e[2] == "goal"]
    first_goal = goals[0] if goals else None

    # Surface Realization
    article = f"""
{lead} in a thrilling match on {datetime.now().strftime('%B %d, %Y')}.

The game started slowly, but {first_goal[1] if first_goal else 'the teams'} broke the deadlock in the {first_goal[0]}th minute.
"""
    for minute, player, event in events:
        if event == "goal":
            article += f"\n{player} ({'home' if player in home else 'away'}) scored in the {minute}'."

    article += f"\n\n{winner} dominated possession and deserved the victory."
    return article.strip()


# === REAL DATA EXAMPLE ===
match = {
    "home_team": "India",
    "away_team": "Australia",
    "home_score": 3,
    "away_score": 1,
    "events": [
        (12, "Virat", "goal"),
        (45, "Smith", "goal"),
        (67, "Rohit", "goal"),
        (89, "Virat", "goal"),
    ],
}

if __name__ == "__main__":
    print("=== AUTOMATED SPORTS NEWS ===\n")
    print(generate_sports_report(match))
