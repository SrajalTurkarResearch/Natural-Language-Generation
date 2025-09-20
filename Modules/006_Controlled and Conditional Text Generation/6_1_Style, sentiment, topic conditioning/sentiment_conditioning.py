# sentiment_conditioning.py
"""
Tutorial on Sentiment Conditioning in NLG
=======================================
For an aspiring scientist: This script teaches sentiment conditioning (emotional tone, e.g., positive vs. negative) in NLG, using your 'car' example. Includes theory, code, visualizations, research directions, applications, and projects.

Theory
------
- Sentiment Conditioning: Controls emotion (positive: 'The car’s awesome'; negative: 'The car’s unreliable').
- Analogy: Sentiment is the 'emoji' of text—😊 for positive, 😣 for negative.
- Math: Adjusts P(word|sentiment). E.g., P(awesome|positive)=0.731.
- Research Relevance: Study sentiment’s impact on trust or ethics in car reviews.

Setup
-----
1. Install: pip install transformers torch numpy matplotlib nltk
2. Run: python sentiment_conditioning.py
3. Note: Uses VADER for sentiment analysis.
"""

import nltk

nltk.download("vader_lexicon")
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# Initialize pipeline and sentiment analyzer
generator = pipeline(
    "text-generation", model="distilgpt2", max_length=50, num_return_sequences=1
)
sia = SentimentIntensityAnalyzer()


# Practical Code: Sentiment Conditioning
def generate_sentiment_review(sentiment):
    """Generate car review with specified sentiment."""
    prompt = f"Write a {sentiment} review of a car’s features:"
    return generator(prompt)[0]["generated_text"]


# Generate examples
positive_text = generate_sentiment_review("positive")
negative_text = generate_sentiment_review("negative")

print("Positive Sentiment:", positive_text)
print("Negative Sentiment:", negative_text)

# Sentiment Analysis with VADER
positive_score = sia.polarity_scores(positive_text)
negative_score = sia.polarity_scores(negative_text)
print("Positive Sentiment Score:", positive_score)
print("Negative Sentiment Score:", negative_score)

# Visualization: Sentiment Scores
sentiments = ["Positive", "Negative"]
scores = [positive_score["compound"], negative_score["compound"]]

plt.bar(sentiments, scores, color=["green", "red"])
plt.xlabel("Sentiment")
plt.ylabel("Compound Score")
plt.title("Sentiment Analysis of Generated Reviews")
plt.savefig("sentiment_scores.png")
plt.close()

# Research Directions
"""
- Study how sentiment in car reviews affects consumer trust.
- Experiment: Generate positive/negative reviews and survey buyers.
- Explore bias in sentiment (e.g., unfair negativity toward brands).
"""

# Applications
"""
- E-commerce: Generate positive car reviews to boost sales.
- Research: Study sentiment’s role in automotive safety perceptions.
- See case_studies.md (Case Study 3: Customer Review Analysis).
"""


# Mini Project: Part of Car Description Generator
def mini_project_sentiment():
    """Test sentiment conditioning for car reviews."""
    sentiments = ["positive", "negative"]
    for sentiment in sentiments:
        text = generate_sentiment_review(sentiment)
        score = sia.polarity_scores(text)["compound"]
        print(f"{sentiment.capitalize()} Output: {text} (Score: {score:.2f})")


print("\nMini Project: Sentiment Conditioning")
mini_project_sentiment()

# Future Directions & Tips
"""
- Future: Develop bias-free sentiment conditioning (2025 ethical NLG trend).
- Tip: Use VADER to validate sentiment in your outputs.
- Gap from Tutorial: Missed interpretability (why models choose emotional words).
"""

# Next Steps
"""
1. Test different sentiment prompts (e.g., 'neutral').
2. Join Hugging Face Discord for NLG tips.
3. Read 2025 sentiment analysis papers on arXiv.
"""
