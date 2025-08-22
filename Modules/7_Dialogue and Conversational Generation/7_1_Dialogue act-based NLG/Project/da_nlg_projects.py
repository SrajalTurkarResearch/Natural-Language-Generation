# da_nlg_projects.py
# Dialogue Act-Based NLG: Mini and Major Projects Module
# Author: Grok, inspired by Alan Turing, Albert Einstein, and Nikola Tesla
# Date: August 22, 2025
# Purpose: A modular Python file with mini and major projects for DA-NLG.

try:
    import nltk
    nltk.download('punkt', quiet=True)
    import spacy
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    from transformers import pipeline
except ImportError as e:
    print(f"Error: {e}. Install: pip install nltk spacy scikit-learn transformers")
    print("Also run: python -m spacy download en_core_web_sm")
    exit()

nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])

def rule_based_nlg(dialogue_act, content):
    """Generate response using templates."""
    templates = {
        "Inform": "The {key} is {value}.",
        "Request": "Please provide the {key}.",
        "Confirm": "Your {key} for {value} is confirmed."
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

def mini_project_weather_chatbot(input_text):
    """Mini Project: Weather Chatbot."""
    if 'weather' in input_text.lower():
        return rule_based_nlg('Inform', {'key': 'weather', 'value': 'sunny, 25°C'})
    return "Sorry, I don't understand."

def major_project_ecommerce_bot(input_text):
    """Major Project: E-Commerce Customer Support Bot."""
    utterances = ["What's the time?", "The time is 3 PM.", "Can you confirm?", "Yes, confirmed."]
    labels = ["Request", "Inform", "Confirm", "Accept"]
    predicted_da = simple_da_classifier(utterances, labels, input_text)
    if predicted_da == 'Request':
        content = {'key': 'order status', 'value': 'shipped on 08/14/2025'}
        try:
            generator = pipeline('text-generation', model='gpt2', device=-1)
            response = generator(f"DA: Inform, Content: Order {content['value']} → ", max_length=50)
            return response[0]['generated_text']
        except Exception as e:
            return f"Error in neural NLG: {e}"
    return "Please clarify your request."

def run_projects():
    """Run both projects."""
    print(f"Weather Chatbot: {mini_project_weather_chatbot('What's the weather like?')}")
    print(f"E-Commerce Bot: {major_project_ecommerce_bot('Where's my order?')}")

if __name__ == "__main__":
    run_projects()