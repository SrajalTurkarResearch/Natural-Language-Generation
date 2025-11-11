# iris_analysis_project.py: Major Project - Analyzing Iris Dataset
#
# Welcome, future researcher! This script is from Section 18 of the NL → Code tutorial,
# where we used a Codex-style prompt to analyze the Iris dataset (a classic in science).
# The prompt is “Analyze Iris dataset: calculate average petal length by species.” This
# shows how NL → Code can automate data analysis for research.
#
# Analogy: It’s like asking a lab assistant to summarize experiment results and make a
# chart. Why this matters: In science, NL → Code can analyze datasets (e.g., climate or
# medical data) faster than manual coding.
#
# Run this to see averages and a plot. Copy into your notebook and ask: “How can I use
# this in my field?”

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

# Calculate average petal length by species
avg_petal = df.groupby("species")["petal length (cm)"].mean()
print("Average Petal Length by Species:")
print(avg_petal)

# Visualize with a bar plot
sns.barplot(x=avg_petal.index, y=avg_petal.values)
plt.title("Average Petal Length by Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Explanation: Groups data by species, computes mean petal length, and plots it.
# Real-World Use: In ecology, you could analyze plant traits to study biodiversity.
# Visual Idea: Bar plot shows petal lengths (setosa short, virginica long).
# Notebook Tip: Try analyzing ‘sepal length’. Ask: “How could I adapt this for my data?”
