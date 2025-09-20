# Linguistic and Cognitive Foundations: Pragmatics (Anaphora and Coherence)
# Theory:
# - Pragmatics studies how context shapes meaning.
# - Anaphora: Words (e.g., pronouns) refer to earlier entities.
#   Analogy: Using nicknames for brevity.
# - Coherence: Logical text flow via anaphora and relations.
#   Analogy: A jigsaw puzzle.
# - Applications: Virtual assistants (anaphora), essay scoring (coherence).
# - Research Direction: Improve anaphora resolution with transformers.
# - Rare Insight: Anaphora resolution can enhance sentiment analysis by tracking entities.

import spacy
import matplotlib.pyplot as plt
import networkx as nx


# Anaphora Resolution with spaCy
def anaphora_resolution(sentence):
    print("\nAnaphora Resolution:")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "PRON":
            print(f"Anaphor: {token.text}, Possible Antecedent: {token.head.text}")


# Coherence Visualization
def coherence_graph(sentences):
    print("\nCoherence Graph:")
    G = nx.DiGraph()
    G.add_nodes_from(sentences)
    G.add_edges_from(
        [
            (sentences[0], sentences[1], {"relation": "Continue"}),
            (sentences[1], sentences[2], {"relation": "Cause"}),
        ]
    )
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    )
    plt.title("Coherence Graph")
    plt.savefig("coherence_graph.png")
    plt.close()
    print("Graph saved as 'coherence_graph.png'.")


# Example
sentence = "Mary bought a book. She read it quickly."
anaphora_resolution(sentence)
sentences = ["I went to the store", "I bought milk", "It was on sale"]
coherence_graph(sentences)

# Notes for Researchers:
# - Use anaphora resolution for context-aware NLP (e.g., chatbots).
# - Study coherence for text generation and evaluation.
# - Explore transformer-based anaphora resolution for accuracy.
