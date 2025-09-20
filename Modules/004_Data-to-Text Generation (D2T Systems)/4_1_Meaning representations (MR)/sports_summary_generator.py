# sports_summary_generator.py
# Tutorial Component: Major Project - Sports Summary Generator with AMR
# Purpose: Generate football match summaries using Abstract Meaning Representation (AMR)
# Theory: AMRs are graph-based MRs that capture complex relationships between concepts (e.g., teams, scores).
#         This project shows how to parse an AMR and generate text in different styles (formal, casual),
#         a critical skill for advanced NLG research.

# For Scientists: AMRs are widely used in NLP research (e.g., ACL conferences). This code helps you
#                 practice parsing and generating from AMRs, preparing you for cutting-edge projects.

# Setup Instructions:
# 1. Install required library: `pip install penman`
# 2. Run this file: `python sports_summary_generator.py`
# 3. Ensure AMR string is valid (example provided below)

import penman
from penman.model import Model


# Function: Generate sports summary from AMR
def sports_amr_to_text(amr_string, style="formal"):
    """
    Convert an AMR to a natural language sports summary.
    AMR Example: (s / score-event :ARG0 (b / Barcelona) :ARG1 (r / Real-Madrid) :ARG2 (s2 / 2-1))
    Theory: AMRs represent meaning as a graph with nodes (concepts) and edges (relations).
            NLG involves parsing the graph (content planning), mapping to a structure (sentence planning),
            and generating text (surface realization).
    """
    # Parse AMR using penman
    graph = penman.decode(amr_string, model=Model())
    team1, team2, score = None, None, None
    for triple in graph.triples:
        if triple[1] == ":ARG0":
            team1 = triple[2].split("/")[1]
        if triple[1] == ":ARG1":
            team2 = triple[2].split("/")[1]
        if triple[1] == ":ARG2":
            score = triple[2].split("/")[1]

    # Generate text based on style
    if style == "formal":
        return f"{team1} defeated {team2} with a score of {score}."
    else:
        return f"{team1} beat {team2} {score}!"


# Real-World Application: Sports apps (e.g., Yahoo Sports) use MRs for automated summaries.
# Research Direction: Extend AMRs to capture dynamic events (e.g., goal times, player actions).
# Tip for Scientists: Use datasets like WebNLG to train AMR parsers for sports domains.

# Test the generator
if __name__ == "__main__":
    # Example AMR
    amr_sports = "(s / score-event :ARG0 (b / Barcelona) :ARG1 (r / Real-Madrid) :ARG2 (s2 / 2-1))"
    print("Formal Summary:", sports_amr_to_text(amr_sports, "formal"))
    print("Casual Summary:", sports_amr_to_text(amr_sports, "casual"))

    # For Your Notebook: Write the AMR and both outputs. Try creating an AMR for a new match.
    # Example: (s / score-event :ARG0 (m / Manchester-United) :ARG1 (c / Chelsea) :ARG2 (s2 / 3-2))

# Future Direction: Integrate with neural NLG models (e.g., GPT-3) for more natural outputs.
# Missing from Tutorial: Handling complex AMRs with nested structures (e.g., multiple events).
# Next Steps: Explore AMR parsing tools (e.g., amrlib) and datasets like DART.
