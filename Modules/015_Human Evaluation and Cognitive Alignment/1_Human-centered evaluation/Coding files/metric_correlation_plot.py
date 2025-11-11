# metric_correlation_plot.py: Visualize Metric-Human Score Correlations
# Theory: Plots help diagnose evaluation gaps; traditional metrics often correlate poorly with human judgments.
# Logic: Use DataFrame for data, seaborn for intuitive scatterplot.
# As Turing might compute, visualizations decode complex patterns in AI evaluations.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data: BLEU scores vs. Human ratings (1-5 scale)
data = pd.DataFrame({"BLEU": [0.2, 0.4, 0.6, 0.8], "Human Score": [2, 3, 4, 5]})

# Create scatterplot
sns.scatterplot(data=data, x="BLEU", y="Human Score")
plt.title("Correlation Between BLEU and Human Scores")
plt.xlabel("BLEU Score")
plt.ylabel("Human Rating (1-5)")
plt.show()

# Math Insight: Compute Pearson r manually
# Means: BLEU=0.5, Human=3.5
# Cov = Σ((x_i - mean_x)(y_i - mean_y)) / (n-1) ≈ 0.2
# Std devs: σ_BLEU ≈ 0.258, σ_Human ≈ 1.291
# r ≈ 0.2 / (0.258 * 1.291) ≈ 0.6 (strong in this example; real often lower)
