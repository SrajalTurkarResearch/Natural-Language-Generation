# Linguistic and Cognitive Foundations: Discourse (RST and Centering)
# Theory:
# - Discourse studies how sentences form coherent texts.
# - Rhetorical Structure Theory (RST): Organizes text into trees with relations (e.g., Cause).
#   Analogy: A recipe with main instructions (nucleus) and supporting steps (satellite).
# - Centering Theory: Tracks entity focus for coherence.
#   Analogy: A spotlight following the main character.
# - Applications: Summarization (RST), chatbots (Centering).
# - Research Direction: Automate RST parsing with deep learning.
# - Rare Insight: Centering can predict reader confusion in real-time dialogues.


def rst_analysis(text):
    print("\nRST Analysis:")
    # Simplified: manually define relation
    rst_relation = {"relation": "Cause", "nucleus": text[1], "satellite": text[0]}
    print(f'Relation: {rst_relation["relation"]}')
    print(f'Nucleus: {rst_relation["nucleus"]}')
    print(f'Satellite: {rst_relation["satellite"]}')


def centering_analysis(sentences):
    print("\nCentering Analysis:")
    for i, sent in enumerate(sentences):
        entities = sent.split()  # Simplified entity extraction
        cf = [e for e in entities if e in ["John", "dog", "him"]]
        cb = None if i == 0 else sentences[i - 1].split()[1]  # Assume second word is Cb
        transition = "None" if i == 0 else ("Continue" if cb in cf else "Shift")
        print(f"Sentence {i+1}: {sent}")
        print(f"Cf: {cf}, Cb: {cb}, Transition: {transition}")


# Coherence Metric (Additional Topic)
def coherence_score(sentences):
    transitions = [
        "Continue" if i > 0 and sentences[i - 1].split()[1] in sent.split() else "Shift"
        for i, sent in enumerate(sentences)
        if i > 0
    ]
    probs = {"Continue": 0.9, "Shift": 0.5}
    score = 1.0
    for t in transitions:
        score *= probs[t]
    print(f"\nCoherence Score: {score:.2f}")


# Example
text = ["I missed the bus.", "I was late for work."]
rst_analysis(text)
sentences = ["John saw a dog.", "The dog barked at him."]
centering_analysis(sentences)
coherence_score(sentences)

# Notes for Researchers:
# - Use RST for summarization or argumentation analysis.
# - Use Centering for dialogue systems.
# - Explore entity grids for advanced coherence metrics.
