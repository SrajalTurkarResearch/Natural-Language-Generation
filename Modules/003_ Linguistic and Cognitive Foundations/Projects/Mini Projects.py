# Linguistic and Cognitive Foundations: Mini Projects
# Theory:
# - Mini projects build practical skills for research.
# - Projects: Dependency parser, centering analysis, text simplification.
# - Applications: Portfolio development, NLP experimentation.
# - Research Direction: Extend projects to real datasets (e.g., news articles).
# - Rare Insight: Small-scale projects can lead to novel research questions.

import spacy


# Mini Project 1: Dependency Parser
def mini_project_dependency(sentence, gold_heads):
    print("\nMini Project 1: Dependency Parser")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc:
        print(f"{token.text} --> {token.dep_} --> {token.head.text}")
    # Ensure gold_heads length matches doc length
    if len(gold_heads) != len(doc):
        print(
            "Error: gold_heads length does not match number of tokens in the sentence."
        )
        print(f"gold_heads length: {len(gold_heads)}, tokens: {len(doc)}")
        return
    correct = sum(1 for i, token in enumerate(doc) if token.head.i == gold_heads[i])
    uas = correct / len(doc) if len(doc) > 0 else 0.0
    print(f"UAS: {uas:.2f}")


# Mini Project 2: Centering Analysis
def mini_project_centering(sentences):
    print("\nMini Project 2: Centering Analysis")
    for i, sent in enumerate(sentences):
        entities = sent.split()
        cf = [e for e in entities if e in ["John", "dog", "him"]]
        cb = None
        if i > 0:
            prev_entities = sentences[i - 1].split()
            cb = prev_entities[1] if len(prev_entities) > 1 else None
        if i == 0:
            transition = "None"
        else:
            transition = "Continue" if cb in cf else "Shift"
        print(f"Sentence {i+1}: {sent}, Cf: {cf}, Cb: {cb}, Transition: {transition}")


# Mini Project 3: Text Simplification
def mini_project_simplification(complex_sent, simple_sent):
    print("\nMini Project 3: Text Simplification")
    nlp = spacy.load("en_core_web_sm")

    def cognitive_load(sent):
        doc = nlp(sent)
        if len(doc) == 0:
            return 0.0
        dep_distance = sum(
            abs(i - token.head.i) for i, token in enumerate(doc) if token.head != token
        ) / len(doc)
        clause_depth = sum(1 for token in doc if token.dep_ == "relcl") + 1
        load = 0.6 * dep_distance + 0.4 * clause_depth
        return load

    print(f"Complex: {complex_sent}, Load: {cognitive_load(complex_sent):.2f}")
    print(f"Simple: {simple_sent}, Load: {cognitive_load(simple_sent):.2f}")


# Example
sentence = "The dog chased the cat."
# spaCy tokenization: ['The', 'dog', 'chased', 'the', 'cat', '.'] (6 tokens)
# gold_heads indices: [2, 2, 2, 4, 2, 2] (example: root is at index 2, period attaches to root)
gold_heads = [2, 2, 2, 4, 2, 2]
mini_project_dependency(sentence, gold_heads)
sentences = ["John saw a dog.", "The dog barked at him."]
mini_project_centering(sentences)
complex_sent = "The book, which Mary bought yesterday, is on the table."
simple_sent = "Mary bought the book yesterday. It is on the table."
mini_project_simplification(complex_sent, simple_sent)

# Notes for Researchers:
# - Use these projects to build a research portfolio.
# - Extend to real datasets for publication-worthy experiments.
# - Explore automation of simplification using transformers.
