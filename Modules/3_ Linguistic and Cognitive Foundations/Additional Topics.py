import spacy
from spacy import displacy

# Linguistic and Cognitive Foundations: Additional Topics for Scientists
# Theory:
# - Cognitive Modeling: Simulate human language processing (e.g., attention, memory) in AI.
# - Evaluation Metrics: Quantify performance in parsing, discourse, and pragmatics.
# - Applications: Enhance NLP models, study language disorders.
# - Research Direction: Develop models simulating working memory limits.
# - Rare Insight: Cognitive modeling can bridge NLP and neuroscience.


# Evaluation Metrics Examples
def dependency_metrics(sentence, gold_heads):
    print("\nDependency Metrics:")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    correct = sum(1 for i, token in enumerate(doc) if token.head.i == gold_heads[i])
    uas = correct / len(doc)
    print(f"UAS: {uas:.2f}")


def coherence_metrics(sentences):
    print("\nCoherence Metrics:")
    transitions = [
        "Continue" if i > 0 and sentences[i - 1].split()[1] in sent.split() else "Shift"
        for i, sent in enumerate(sentences)
        if i > 0
    ]
    probs = {"Continue": 0.9, "Shift": 0.5}
    score = 1.0
    for t in transitions:
        score *= probs[t]
    print(f"Coherence Score: {score:.2f}")


# Example
sentence = "The dog chased the cat."
gold_heads = [2, 2, 2, 4, 2]
dependency_metrics(sentence, gold_heads)
sentences = ["John saw a dog.", "The dog barked at him."]
coherence_metrics(sentences)

# Notes for Researchers:
# - Explore cognitive modeling with transformers (e.g., attention mechanisms).
# - Develop new metrics for coherence in multilingual texts.
# - Study language processing in neurological disorders.
