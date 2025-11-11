# metric_plot.py: Visualization of NLG metrics.
# This creates a bar plot to compare quant metrics visually,
# aiding in research presentations or exploratory data analysis.
# As Einstein visualized relativity, plot data to see patterns.

import matplotlib.pyplot as plt

# Sample metrics and scores (replace with your own calculations).
metrics = ["BLEU", "ROUGE", "METEOR"]
scores = [0.36, 0.667, 0.5]  # Example values from a study.

# Create a bar plot.
plt.bar(metrics, scores, color=["blue", "green", "red"])
plt.title("NLG Evaluation Metrics Comparison")
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.ylim(0, 1)  # Standardize scale for scores 0-1.
plt.show()

# Tip for Aspiring Researchers: Use this in reports to illustrate quant findings alongside qual insights.
