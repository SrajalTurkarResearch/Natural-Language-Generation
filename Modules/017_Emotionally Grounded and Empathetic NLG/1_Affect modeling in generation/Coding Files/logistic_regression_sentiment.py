from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def train_sentiment_classifier():
    """
    Train a logistic regression model for sentiment analysis on a small dataset.
    """
    # Sample dataset
    texts = ["I love this!", "This is awful.", "It's okay.", "I'm so happy!"]
    labels = [1, 0, 0, 1]  # 1 = positive, 0 = negative/neutral

    # Convert text to numerical features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts).toarray()

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X, labels)

    # Test prediction
    test_text = ["I'm thrilled!"]
    X_test = vectorizer.transform(test_text).toarray()
    prediction = model.predict_proba(X_test)[0][1]  # Probability of positive
    print(f"Text: {test_text[0]}")
    print(f"Probability of positive sentiment: {prediction:.3f}")


if __name__ == "__main__":
    train_sentiment_classifier()
