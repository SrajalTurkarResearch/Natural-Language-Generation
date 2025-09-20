# topic_conditioning.py
"""
Tutorial on Topic Conditioning in NLG
====================================
For an aspiring scientist: This script teaches topic conditioning (subject focus, e.g., vehicles) in NLG, using your 'car' example. Includes theory, code, visualizations, research directions, applications, and projects.

Theory
------
- Topic Conditioning: Ensures text stays on subject (vehicles: 'The car’s engine is hybrid'; not cooking: 'The cake’s yummy').
- Analogy: Topic is a 'GPS' keeping text on the vehicle route.
- Math: Adjusts P(word|topic). E.g., P(car|vehicles)=0.4, P(herb|vehicles)=0.01.
- Research Relevance: Focus NLG on research domains like car dynamics.

Setup
-----
1. Install: pip install transformers torch wordcloud matplotlib nltk
2. Run: python topic_conditioning.py
3. Note: Uses wordcloud for visualization.
"""

import nltk

nltk.download("vader_lexicon")
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize pipeline
generator = pipeline(
    "text-generation", model="distilgpt2", max_length=50, num_return_sequences=1
)


# Practical Code: Topic Conditioning
def generate_topic_description(topic):
    """Generate text with specified topic."""
    prompt = f"Write about a car’s engine in the context of {topic}:"
    return generator(prompt)[0]["generated_text"]


# Generate examples
vehicle_text = generate_topic_description("vehicles")
cooking_text = generate_topic_description("cooking")

print("Vehicles Topic:", vehicle_text)
print("Cooking Topic (Incorrect):", cooking_text)

# Visualization: Word Cloud for Vehicles Topic
word_freq = {"car": 0.4, "truck": 0.3, "engine": 0.2, "herb": 0.01}
wordcloud = WordCloud(
    width=400, height=200, background_color="white"
).generate_from_frequencies(word_freq)
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Vehicles Topic Word Cloud")
plt.savefig("vehicles_wordcloud.png")
plt.close()

# Research Directions
"""
- Study topic coherence in vehicle-focused NLG for autonomous driving.
- Experiment: Generate vehicle vs. non-vehicle text and measure coherence.
- Explore low-resource language topic conditioning.
"""

# Applications
"""
- Navigation: Generate vehicle-focused driving instructions.
- Research: Automate vehicle dynamics reports.
- See case_studies.md (Case Study 4: Driver-AI Interaction).
"""


# Mini Project: Part of Car Description Generator
def mini_project_topic():
    """Test topic conditioning for car descriptions."""
    topics = ["vehicles", "cooking"]
    for topic in topics:
        print(f"{topic.capitalize()} Output:", generate_topic_description(topic))


print("\nMini Project: Topic Conditioning")
mini_project_topic()

# Future Directions & Tips
"""
- Future: Multimodal topic conditioning (text + car images).
- Tip: Check topic relevance manually to spot drift.
- Gap from Tutorial: Missed multilingual topic conditioning (e.g., Hindi car reviews).
"""

# Next Steps
"""
1. Try topics like 'safety' or 'performance'.
2. Read LDA topic modeling papers for deeper understanding.
3. Share word clouds on X to showcase your work.
"""
