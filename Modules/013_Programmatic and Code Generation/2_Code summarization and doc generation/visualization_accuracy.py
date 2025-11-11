# visualization_accuracy.py
# Purpose: Plots a bar chart comparing NLG model accuracy (BLEU scores).
# Why: Scientists use plots to compare tools, like choosing the best microscope.
# Requires: matplotlib (run setup.py first).

import matplotlib.pyplot as plt

# Fake data for demo (real BLEU scores need model outputs)
models = ["Rules", "Stats", "RNN", "Transformer"]
bleu_scores = [0.4, 0.5, 0.7, 0.9]  # Higher is better

# Create bar plot
plt.bar(models, bleu_scores, color="skyblue")
plt.xlabel("Model Type")
plt.ylabel("BLEU Score")
plt.title("Summary Quality by Model Type")
plt.show()

# Why this matters: Shows Transformers (2025) are best, like top lab equipment.
# For science: Helps pick the right NLG tool for research code.
# Try it: Add a fake model (e.g., 'CodeT5', 0.95) and replot.
