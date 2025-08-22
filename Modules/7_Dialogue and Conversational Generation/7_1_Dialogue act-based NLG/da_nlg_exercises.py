# da_nlg_exercises.py
# Dialogue Act-Based NLG: Practical Exercises Module
# Author: Grok, inspired by Alan Turing, Albert Einstein, and Nikola Tesla
# Date: August 22, 2025
# Purpose: A modular Python file with practical exercises for DA-NLG learning.

try:
    import nltk

    nltk.download("punkt", quiet=True)
    import spacy
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError as e:
    print(
        f"Error: {e}. Install: pip install nltk spacy scikit-learn matplotlib networkx"
    )
    exit()

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def rule_based_nlg(dialogue_act, content):
    """Generate response using templates."""
    templates = {
        "Inform": "The {key} is {value}.",
        "Request": "Please provide the {key}.",
        "Confirm": "Your {key} for {value} is confirmed.",
    }
    return templates.get(dialogue_act, "Sorry, I don't understand.").format(**content)


def simple_da_classifier(utterances, labels, test_utterance):
    """Train a Naive Bayes classifier for dialogue acts."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(utterances)
    clf = MultinomialNB()
    clf.fit(X, labels)
    pred = clf.predict(vectorizer.transform([test_utterance]))
    return pred[0]


def exercise_1():
    """Extend DA classifier with more data."""
    utterances = [
        "What's the time?",
        "The time is 3 PM.",
        "Can you confirm?",
        "Yes, confirmed.",
        "Tell me about the event.",
        "Thanks!",
    ]
    labels = ["Request", "Inform", "Confirm", "Accept", "Request", "Thank"]
    test = "What's the event about?"
    print(
        f"Exercise 1 - Predicted DA: {simple_da_classifier(utterances, labels, test)}"
    )


def exercise_2():
    """Visualize DA transition probabilities."""
    G = nx.DiGraph()
    G.add_edges_from(
        [("Request", "Inform", {"weight": 0.6}), ("Inform", "Confirm", {"weight": 0.3})]
    )
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 5))
    nx.draw(
        G, pos, with_labels=True, node_color="lightgreen", node_size=2000, font_size=12
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("DA Transition Probabilities")
    plt.savefig("da_transitions.png")
    plt.close()
    print("Exercise 2 - Saved: da_transitions.png")


def exercise_3():
    """Implement a template-based NLG."""
    content = {"key": "flight", "value": "confirmed for 5 PM"}
    print(f"Exercise 3 - NLG Response: {rule_based_nlg('Confirm', content)}")


def run_exercises():
    """Run all exercises."""
    exercise_1()
    exercise_2()
    exercise_3()


if __name__ == "__main__":
    run_exercises()
