# Linguistic and Cognitive Foundations: Cognitive Constraints in Generation
# Theory:
# - Cognitive constraints (working memory, attention, cognitive load) affect language processing.
# - Analogy: Cooking with limited counter space, handling a few ingredients (words).
# - Working Memory: ~7 ± 2 items (Miller’s Law).
# - Cognitive Load: Higher for complex sentences.
# - Applications: Text simplification, AI design (transformers).
# - Research Direction: Model working memory in neural text generators.
# - Rare Insight: Emotional context may influence attention in language processing.

import spacy


# Measure Cognitive Load
def cognitive_load(sentence):
    print("\nCognitive Load Analysis:")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    dep_distance = sum(
        abs(i - token.head.i) for i, token in enumerate(doc) if token.head != token
    ) / len(doc)
    clause_depth = sum(1 for token in doc if token.dep_ in ["relcl"]) + 1  # Simplified
    load = 0.6 * dep_distance + 0.4 * clause_depth
    print(f"Sentence: {sentence}")
    print(f"Dependency Distance: {dep_distance:.2f}")
    print(f"Clause Depth: {clause_depth}")
    print(f"Cognitive Load: {load:.2f}")


# Example
complex_sent = "The book, which Mary bought yesterday, is on the table."
simple_sent = "Mary bought the book yesterday. It is on the table."
cognitive_load(complex_sent)
cognitive_load(simple_sent)

# Notes for Researchers:
# - Use cognitive load metrics for text simplification.
# - Explore attention mechanisms in transformers for cognitive modeling.
# - Study emotional influences on language processing.
