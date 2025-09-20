# Linguistic and Cognitive Foundations: Grammar (Dependency and Constituency)
# Theory:
# - Grammar defines how words form sentences.
# - Dependency Grammar: Focuses on word-to-word relationships, with the verb as the root.
#   Analogy: A mobile with the verb as the main hook and words as ornaments.
# - Constituency Grammar: Breaks sentences into nested phrases (e.g., NP, VP).
#   Analogy: A family tree with the sentence as the ancestor.
# - Applications: Machine translation (dependency), grammar checkers (constituency).
# - Research Direction: Develop hybrid parsers for multilingual NLP.
# - Rare Insight: Dependency parsing is robust for low-resource languages due to flexibility.

import spacy
from spacy import displacy
import nltk
from nltk import CFG, ChartParser
import matplotlib.pyplot as plt


# Dependency Parsing with spaCy
def dependency_parsing(sentence):
    print("\nDependency Parsing:")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc:
        print(
            f"{token.text} --> {token.dep_} --> {token.head.text}"
        )  # token.dep_ is the dependency relation of the token (e.g. nsubj, dobj, etc.) token.head.text is the text of the head of the token
    # Serve visualization (save to HTML for external viewing)
    html = displacy.render(doc, style="dep", options={"compact": True}, jupyter=False)
    with open("dependency_tree.html", "w") as f:
        f.write(html)
    print("Dependency tree saved as 'dependency_tree.html'.")


# Constituency Parsing with NLTK
def constituency_parsing(sentence):
    print("\nConstituency Parsing:")
    grammar = CFG.fromstring(
        """
        S -> NP VP
        NP -> Det Adj N | Det N
        VP -> V NP
        Det -> 'The' | 'the'
        Adj -> 'quick' | 'small'
        N -> 'dog' | 'cat'
        V -> 'chased'
    """
    )
    parser = ChartParser(grammar)
    words = sentence.split()
    for tree in parser.parse(words):
        print(tree)
        tree.pretty_print()


# Evaluation Metric: Unlabeled Attachment Score (UAS)
def compute_uas(sentence, gold_heads):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    correct = sum(1 for i, token in enumerate(doc) if token.head.i == gold_heads[i])
    uas = correct / len(doc)
    print(f"\nUAS: {uas:.2f}")


# Example
sentence = "The quick dog chased the small cat"
dependency_parsing(sentence)
constituency_parsing(sentence)
gold_heads = [2, 2, 3, 3, 5, 6, 3]  # Indices of heads for the sentence
compute_uas(sentence, gold_heads)

# Notes for Researchers:
# - Use dependency parsing for flexible word-order languages (e.g., Russian).
# - Use constituency parsing for strict word-order languages (e.g., English).
# - Explore hybrid models combining both for robust NLP parsing.
