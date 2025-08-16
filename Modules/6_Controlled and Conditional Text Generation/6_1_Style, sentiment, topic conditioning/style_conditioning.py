# style_conditioning.py
"""
Tutorial on Style Conditioning in NLG
====================================
For an aspiring scientist: This script teaches style conditioning (how text is said, e.g., formal vs. casual) in NLG, using your 'car' example. It includes theory, code, visualizations, research directions, applications, and projects. Run this to generate car descriptions and explore research ideas.

Theory
------
- Style Conditioning: Controls tone/structure (formal: 'The vehicle is efficient'; casual: 'This car’s cool').
- Analogy: Style is the 'voice' of your AI—formal like a professor, casual like a friend.
- Math: Adjusts P(word|style). E.g., P(efficient|formal)=0.7, P(cool|formal)=0.3.
- Research Relevance: Tailor outputs for scientific reports (formal) or outreach (casual).

Setup
-----
1. Install: pip install transformers torch numpy matplotlib nltk
2. Run: python style_conditioning.py
3. Note: Uses distilgpt2 (lightweight). For better results, try gpt2-medium on GPU/Colab.
"""

import nltk

nltk.download("vader_lexicon")
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

# Initialize text generation pipeline
generator = pipeline(
    "text-generation", model="distilgpt2", max_length=50, num_return_sequences=1
)


# Practical Code: Style Conditioning
def generate_style_description(style):
    """Generate car description with specified style."""
    prompt = f"Write a {style} description of a car’s performance:"
    return generator(prompt)[0]["generated_text"]


# Generate examples
formal_text = generate_style_description("formal")
casual_text = generate_style_description("casual")

print("Formal Style:", formal_text)
print("Casual Style:", casual_text)

# Visualization: Probability of word choices
words = ["efficient", "cool"]
formal_probs = [0.7, 0.3]
casual_probs = [0.3, 0.7]

x = np.arange(len(words))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width / 2, formal_probs, width, label="Formal")
ax.bar(x + width / 2, casual_probs, width, label="Casual")
ax.set_xlabel("Words")
ax.set_ylabel("Probability")
ax.set_title("Style Conditioning: Word Probabilities")
ax.set_xticks(x)
ax.set_xticklabels(words)
ax.legend()
plt.savefig("style_probs.png")  # Save for your notes
plt.close()

# Research Directions
"""
- Study how formal vs. casual style affects trust in car safety reports.
- Experiment: Generate reports and survey readers.
- Read 2025 NLG papers on arXiv for style control trends.
"""

# Applications
"""
- Automotive: Formal for manuals, casual for ads ('Zoom into adventure!').
- Research: Generate formal car physics reports or casual STEM blogs.
- See case_studies.md (Case Study 1: Automotive Advertising).
"""


# Mini Project: Part of Car Description Generator
def mini_project_style():
    """Test style conditioning for car descriptions."""
    styles = ["formal", "casual"]
    for style in styles:
        print(f"{style.capitalize()} Output:", generate_style_description(style))


print("\nMini Project: Style Conditioning")
mini_project_style()

# Future Directions & Tips
"""
- Future: Explore style control vectors for precise NLG (2025 trend).
- Tip: Experiment with prompts (e.g., 'ultra-formal') to see variations.
- Gap from Tutorial: Missed fine-tuning for custom styles (try DistilBERT fine-tuning).
"""

# Next Steps
"""
1. Tweak prompts for different styles (e.g., 'technical', 'poetic').
2. Share results on X or GitHub to build your research profile.
3. Read 'Attention is All You Need' (Transformer paper).
"""
