# nlg_pipeline_project.py
# Project: Implement a full Classical NLG Pipeline with manual rule logic and planners
# Generates a soccer game summary using rule-based logic
# Designed for beginners aiming to become scientists and researchers

# Install required libraries: pip install nltk pandas matplotlib
import nltk
import pandas as pd
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Sample soccer game data (as a pandas DataFrame for realism)
data = pd.DataFrame(
    {
        "match_id": [1],
        "team_a": ["Lions"],
        "team_b": ["Tigers"],
        "score_a": [3],
        "score_b": [2],
        "key_player": ["Alex"],
        "player_goals": [2],
        "key_moment": ["90th minute goal"],
        "match_date": ["2025-08-04"],
    }
)


# Stage 1: Content Determination
def content_determination(data, match_id):
    """Select relevant data based on manual rules."""
    # Rule: Select match details, score, key player, and key moment if score difference exists
    row = data[data["match_id"] == match_id].iloc[0]
    if row["score_a"] != row["score_b"]:  # Ensure a winner for summary
        selected = {
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "score": f"{row['score_a']}-{row['score_b']}",
            "key_player": row["key_player"],
            "player_goals": row["player_goals"],
            "key_moment": row["key_moment"],
        }
    else:
        selected = {
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "score": f"{row['score_a']}-{row['score_b']}",
        }
    return selected


# Stage 2: Document Planning
def document_planning(selected_data):
    """Create a structured plan using manual rules."""
    # Rule: Structure as intro (teams), body (score, player), conclusion (key moment or draw)
    plan = {
        "introduction": f"{selected_data['team_a']} vs. {selected_data['team_b']}",
        "body": {
            "score": selected_data["score"],
            "player": (
                f"{selected_data.get('key_player', 'No key player')} scored {selected_data.get('player_goals', 0)} goals"
                if "key_player" in selected_data
                else None
            ),
        },
        "conclusion": selected_data.get("key_moment", "The match ended in a draw."),
    }
    return plan


# Stage 3: Microplanning
def microplanning(doc_plan):
    """Define linguistic choices with manual rules."""
    # Rules:
    # - Lexical choice: Use "defeated" for winner, "drew" for tie
    # - Aggregation: Combine score and player into one sentence
    # - Tone: Engaging for sports fans
    score = doc_plan["body"]["score"]
    team_a, team_b = doc_plan["introduction"].split(" vs. ")
    score_a, score_b = map(int, score.split("-"))

    if score_a > score_b:
        main_sentence = f"{team_a} defeated {team_b} {score}"
        if doc_plan["body"]["player"]:
            main_sentence += f", with {doc_plan['body']['player'].lower()}."
    else:
        main_sentence = f"{team_a} and {team_b} drew {score}."

    conclusion = (
        f"The match was decided by a {doc_plan['conclusion']}."
        if score_a > score_b
        else doc_plan["conclusion"]
    )
    return {"main_sentence": main_sentence, "conclusion": conclusion}


# Stage 4: Surface Realization
def surface_realization(microplanned):
    """Generate grammatically correct sentences using manual rules."""
    # Rule: Ensure subject-verb-object structure and proper punctuation
    main_sentence = microplanned["main_sentence"]
    tokens = nltk.word_tokenize(main_sentence)
    tagged = nltk.pos_tag(tokens)
    has_verb = any(tag.startswith("VB") for word, tag in tagged)
    if has_verb:
        return f"{main_sentence} {microplanned['conclusion']}"
    return "Error: Sentence lacks a verb."


# Stage 5: Post-Processing
def post_processing(realized):
    """Polish text with manual rules."""
    # Rules: Capitalize proper nouns, fix number words, ensure proper spacing
    polished = realized.replace("alex", "Alex")  # Capitalize player name
    polished = polished.replace("2", "two")  # Convert numbers to words for style
    polished = polished.replace(".,", ".")  # Fix punctuation
    return polished


# Stage 6: Output
def output(final_text, data, match_id):
    """Deliver the final text and visualize scores."""
    print("Soccer Game Summary:")
    print(final_text)
    # Visualize scores
    row = data[data["match_id"] == match_id].iloc[0]
    plt.bar(
        [row["team_a"], row["team_b"]],
        [row["score_a"], row["score_b"]],
        color=["blue", "orange"],
    )
    plt.title("Game Scores")
    plt.ylabel("Goals")
    plt.show()


# Run the pipeline
def run_pipeline(data, match_id):
    """Execute the full NLG pipeline."""
    print("Running Classical NLG Pipeline...")
    selected_data = content_determination(data, match_id)
    print("Selected Data:", selected_data)

    doc_plan = document_planning(selected_data)
    print("Document Plan:", doc_plan)

    microplanned = microplanning(doc_plan)
    print("Microplanned Content:", microplanned)

    realized = surface_realization(microplanned)
    print("Realized Text:", realized)

    final_text = post_processing(realized)
    print("Final Text:", final_text)

    output(final_text, data, match_id)
    return final_text


if __name__ == "__main__":
    final_summary = run_pipeline(data, 1)
    print("Project Completed!")
