# combined_conditioning_projects.py
"""
Tutorial on Combined Conditioning and Projects in NLG
=================================================
For an aspiring scientist: This script integrates style, sentiment, and topic conditioning, with mini and major projects for car-related NLG. Includes theory, code, research directions, applications, and projects.

Theory
------
- Combined Conditioning: Integrates style, sentiment, topic (e.g., formal, positive, vehicles: 'The vehicle is remarkably efficient').
- Analogy: Like mixing paint—style (color), sentiment (brightness), topic (subject).
- Math: P(word|style,sentiment,topic) ≈ P(style)*P(sentiment)*P(topic).
- Research Relevance: Create precise, ethical NLG for automotive research.

Setup
-----
1. Install: pip install transformers torch numpy matplotlib nltk pandas scikit-learn
2. Run: python combined_conditioning_projects.py
3. Note: Includes mini and major project implementations.
"""

import nltk

nltk.download("vader_lexicon")
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

# Initialize pipeline and sentiment analyzer
generator = pipeline(
    "text-generation", model="distilgpt2", max_length=50, num_return_sequences=1
)
sia = SentimentIntensityAnalyzer()


# Practical Code: Combined Conditioning
def generate_combined_description(style, sentiment, topic):
    """Generate car description with style, sentiment, topic."""
    prompt = f"Write a {style} {sentiment} description of a car’s features in the context of {topic}:"
    return generator(prompt)[0]["generated_text"]


# Generate example
combined_text = generate_combined_description("formal", "positive", "vehicles")
print("Combined (Formal, Positive, Vehicles):", combined_text)


# Mini Project: Car Description Generator
def mini_project_combined():
    """Generate car descriptions with different condition combinations."""
    combinations = [
        ("formal", "positive", "vehicles"),
        ("casual", "negative", "vehicles"),
    ]
    results = []
    for style, sentiment, topic in combinations:
        text = generate_combined_description(style, sentiment, topic)
        score = sia.polarity_scores(text)["compound"]
        results.append(
            {
                "Style": style,
                "Sentiment": sentiment,
                "Topic": topic,
                "Text": text,
                "Score": score,
            }
        )
        print(
            f"{style.capitalize()}/{sentiment.capitalize()}/{topic.capitalize()}: {text} (Score: {score:.2f})"
        )
    return pd.DataFrame(results)


print("\nMini Project: Car Description Generator")
df_mini = mini_project_combined()


# Major Project: Biased vs. Unbiased Car Reviews
def major_project_biased_unbiased():
    """Generate and compare biased vs. unbiased car reviews."""
    biased_prompt = "Write a negative review unfairly targeting a specific car brand:"
    unbiased_prompt = "Write a neutral review of a car’s features:"
    biased_text = generator(biased_prompt)[0]["generated_text"]
    unbiased_text = generator(unbiased_prompt)[0]["generated_text"]
    biased_score = sia.polarity_scores(biased_text)["compound"]
    unbiased_score = sia.polarity_scores(unbiased_text)["compound"]
    print("Biased Review:", biased_text, f"(Score: {biased_score:.2f})")
    print("Unbiased Review:", unbiased_text, f"(Score: {unbiased_score:.2f})")

    # Visualization: Sentiment Comparison
    plt.bar(
        ["Biased", "Unbiased"], [biased_score, unbiased_score], color=["red", "blue"]
    )
    plt.xlabel("Review Type")
    plt.ylabel("Sentiment Score")
    plt.title("Biased vs. Unbiased Car Reviews")
    plt.savefig("review_sentiment.png")
    plt.close()


print("\nMajor Project: Biased vs. Unbiased Reviews")
major_project_biased_unbiased()

# Research Directions
"""
- Study how combined conditioning affects trust in car safety reports.
- Experiment: Generate reviews with varying conditions and survey consumers.
- Explore ethical NLG to reduce bias in automotive applications.
"""

# Applications
"""
- Industry: Generate tailored car ads or reports.
- Research: Simulate driver-AI dialogues or automate vehicle studies.
- See case_studies.md (Case Study 2: Car Safety Reports, Case Study 4: Driver-AI Interaction).
"""

# Future Directions & Tips
"""
- Future: Multimodal NLG (text + car images).
- Tip: Validate outputs with metrics like BLEU or coherence.
- Gap from Tutorial: Missed human-in-the-loop feedback for refining conditioning.
"""

# Next Steps
"""
1. Run mini project with new combinations (e.g., poetic/positive/safety).
2. Submit major project findings to arXiv.
3. Learn PyTorch for custom NLG models.
"""
