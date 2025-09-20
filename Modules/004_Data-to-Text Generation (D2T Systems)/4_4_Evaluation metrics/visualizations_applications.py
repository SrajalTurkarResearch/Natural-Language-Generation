# visualizations_applications.py
# A beginner-friendly tutorial on NLG evaluation metrics: Visualizations and Applications
# Includes code for visualizing metric comparisons and real-world use cases

"""
Visualizations
=============
Visualizations help compare NLG models. Example: Bar plot comparing BLEU, ROUGE, and BERTScore for two models.

Real-World Applications
======================
1. Chatbots: Evaluate responses for customer support (e.g., relevance, fluency).
   Example: A retail chatbot answering "What’s the return policy?"
2. Summarization: Assess news or document summaries (e.g., ROUGE for content).
   Example: Summarizing a news article about a hurricane.
3. Machine Translation: Compare translations to references (e.g., BLEU).
4. Creative Writing: Measure diversity in stories (e.g., Distinct-n).

Instructions to Run
==================
1. Install libraries: `pip install matplotlib seaborn pandas`
2. Save as `visualizations_applications.py`
3. Run: `python visualizations_applications.py`
4. Outputs a bar plot comparing two models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data for two models
data = {
    "Model": ["Model A", "Model A", "Model A", "Model B", "Model B", "Model B"],
    "Metric": ["BLEU", "ROUGE-1", "BERTScore", "BLEU", "ROUGE-1", "BERTScore"],
    "Score": [0.75, 0.80, 0.90, 0.65, 0.85, 0.88],
}
df = pd.DataFrame(data)

# Bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="Score", hue="Metric", data=df)
plt.title("Model Comparison: Evaluation Metrics")
plt.ylabel("Score")
plt.show()

if __name__ == "__main__":
    print("NLG Evaluation Tutorial: Visualizations and Applications")
    print("Run complete. Check the bar plot.")
    print("Next: Run `projects_research.py` for projects and research directions.")
