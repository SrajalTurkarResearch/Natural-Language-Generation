# da_nlg_code_guides.py
# Dialogue Act-Based NLG: Practical Code Guides Module
# Author: Grok, inspired by Alan Turing, Albert Einstein, and Nikola Tesla
# Date: August 22, 2025
# Purpose: A modular Python file with executable code for DA classification and NLG.

try:
    import nltk

    nltk.download("punkt", quiet=True)
    import spacy
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    from transformers import pipeline
except ImportError as e:
    print(f"Error: {e}. Install: pip install nltk spacy scikit-learn transformers")
    print("Also run: python -m spacy download en_core_web_sm")
    exit()

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def simple_da_classifier(utterances, labels, test_utterance):
    """Train a Naive Bayes classifier for dialogue acts."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(utterances)
    clf = MultinomialNB()
    clf.fit(X, labels)
    pred = clf.predict(vectorizer.transform([test_utterance]))
    return pred[0]


def rule_based_nlg(dialogue_act, content):
    """Generate response using templates."""
    templates = {
        "Inform": "The {key} is {value}.",
        "Request": "Please provide the {key}.",
        "Confirm": "Your {key} for {value} is confirmed.",
    }
    return templates.get(dialogue_act, "Sorry, I don't understand.").format(**content)


def neural_nlg(prompt):
    """Generate response using GPT-2."""
    try:
        generator = pipeline("text-generation", model="gpt2", device=-1)
        response = generator(prompt, max_length=50, num_return_sequences=1)
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error in neural NLG: {e}"


def run_code_examples():
    """Run example code for DA classification and NLG."""
    # DA Classification
    utterances = [
        "What's the time?",
        "The time is 3 PM.",
        "Can you confirm?",
        "Yes, confirmed.",
    ]
    labels = ["Request", "Inform", "Confirm", "Accept"]
    test_utterance = "Is it raining?"
    print(
        f"DA Classification: {simple_da_classifier(utterances, labels, test_utterance)}"
    )

    # Rule-Based NLG
    content = {"key": "meeting time", "value": "2 PM"}
    print(f"Rule-Based NLG: {rule_based_nlg('Inform', content)}")

    # Neural NLG
    prompt = "DA: Inform, Content: Meeting at 2 PM → "
    print(f"Neural NLG: {neural_nlg(prompt)}")


if __name__ == "__main__":
    run_code_examples()
