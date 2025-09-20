# Paragraph Generator with Discourse Structure and Rhetorical Intent
# Theory:
# - Discourse Structure: Organizes sentences into a coherent paragraph using relations like Cause, Elaboration, Contrast (inspired by Rhetorical Structure Theory [RST]).
#   Analogy: A recipe where sentences (ingredients) are linked logically to form a dish (paragraph).
# - Rhetorical Intent: Guides the paragraph's purpose (e.g., persuasion, explanation, narration).
#   Analogy: A speech tailored to convince, inform, or tell a story.
# - Centering Theory: Ensures coherence by maintaining focus on key entities (e.g., 'John', 'dog').
# - Applications: Automated content generation, educational tools, chatbots.
# - Research Direction: Integrate neural models (e.g., transformers) for dynamic discourse planning.
# - Rare Insight: Discourse-aware generators can enhance user trust in AI by mimicking human-like intent.

import random
import networkx as nx
import matplotlib.pyplot as plt
import spacy

# Initialize spaCy for entity tracking
nlp = spacy.load("en_core_web_sm")

# Define discourse relations and rhetorical intents
relations = ["Cause", "Elaboration", "Contrast"]
intents = {
    "persuasion": ["urge", "highlight benefits", "address objections"],
    "explanation": ["state fact", "explain reason", "provide example"],
    "narration": ["set scene", "describe action", "conclude event"],
}


# Paragraph Generator
def generate_paragraph(entities, actions, intent="persuasion"):
    """
    Generate a coherent paragraph with discourse structure and rhetorical intent.
    Args:
        entities (list): Main entities (e.g., ['John', 'dog']).
        actions (list): Actions for sentences (e.g., ['saw a dog', 'adopted it']).
        intent (str): Rhetorical intent (persuasion, explanation, narration).
    Returns:
        list: Generated sentences, discourse relations, and entity focus.
    """
    print(f"\nGenerating Paragraph (Intent: {intent})")
    sentences = []
    discourse_structure = []
    entity_focus = [entities[0]]  # Initial focus (Cb)

    # Generate sentences based on intent
    intent_steps = intents[intent]
    for i, step in enumerate(intent_steps[:3]):  # Limit to 3 sentences for simplicity
        if i == 0:
            # First sentence: Introduce main entity
            sentence = f"{entities[0]} {actions[0]}."
            sentences.append(sentence)
        elif i == 1:
            # Second sentence: Maintain focus or shift
            relation = random.choice(relations)
            sentence = (
                f"It {actions[1]} {entities[1]}."
                if relation != "Contrast"
                else f"But {entities[1]} {actions[1]}."
            )
            sentences.append(sentence)
            discourse_structure.append((sentences[i - 1], sentences[i], relation))
            entity_focus.append(entities[1] if relation != "Contrast" else entities[0])
        else:
            # Third sentence: Conclude or elaborate
            relation = (
                "Elaboration" if intent == "explanation" else random.choice(relations)
            )
            sentence = f"This {random.choice(['was exciting', 'helped a lot', 'changed everything'])}."
            sentences.append(sentence)
            discourse_structure.append((sentences[i - 1], sentences[i], relation))
            entity_focus.append(entity_focus[-1])  # Maintain focus

    return sentences, discourse_structure, entity_focus


# Centering Analysis for Coherence
def centering_analysis(sentences, entity_focus):
    print("\nCentering Analysis:")
    for i, sent in enumerate(sentences):
        cf = [e for e in sent.split() if e in entity_focus]
        cb = entity_focus[i]
        transition = (
            "None" if i == 0 else ("Continue" if cb == entity_focus[i - 1] else "Shift")
        )
        print(f"Sentence {i+1}: {sent}, Cf: {cf}, Cb: {cb}, Transition: {transition}")


# Visualize Discourse Structure
def visualize_discourse(sentences, discourse_structure):
    print("\nVisualizing Discourse Structure:")
    G = nx.DiGraph()
    G.add_nodes_from(sentences)
    G.add_edges_from(
        [(s1, s2, {"relation": rel}) for s1, s2, rel in discourse_structure]
    )
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    )
    plt.title(f"Discourse Structure (Intent: {intent})")
    plt.savefig("discourse_structure.png")
    plt.close()
    print("Graph saved as 'discourse_structure.png'.")


# Example
entities = ["John", "dog"]
actions = ["saw a stray dog", "adopted it"]
intent = "narration"
sentences, discourse_structure, entity_focus = generate_paragraph(
    entities, actions, intent
)
print("Generated Paragraph:", " ".join(sentences))
centering_analysis(sentences, entity_focus)
visualize_discourse(sentences, discourse_structure)


# Mini Project: Extend the Generator
# Task: Add a new intent (e.g., 'education') and generate a 4-sentence paragraph.
def mini_project_extended_generator():
    print("\nMini Project: Extended Paragraph Generator")
    new_intent = "education"
    intents[new_intent] = [
        "introduce topic",
        "explain concept",
        "give example",
        "summarize",
    ]
    entities = ["Teacher", "students"]
    actions = ["introduced AI", "explained its benefits"]
    sentences, discourse_structure, entity_focus = generate_paragraph(
        entities, actions + ["used a chatbot", "learned faster"], new_intent
    )
    print("Extended Paragraph:", " ".join(sentences))
    centering_analysis(sentences, entity_focus)
    visualize_discourse(sentences, discourse_structure)


# Run Mini Project
mini_project_extended_generator()

# Notes for Researchers:
# - Applications: Automated storytelling, educational content, persuasive ads.
# - Research Direction: Use transformers (e.g., GPT-3) for dynamic discourse planning.
# - Rare Insight: Intent-driven generation can enhance user engagement in chatbots.
# - Next Steps: Test with real datasets (e.g., news articles) and evaluate coherence.
