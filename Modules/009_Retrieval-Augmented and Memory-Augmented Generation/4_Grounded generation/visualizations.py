# visualizations.py: Plots and Diagrams for Grounded NLG
# This file creates plots and describes diagrams to sketch.
# Run with matplotlib and seaborn installed: pip install matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns

# 3.1 Plot: Accuracy vs. Grounding Strength
# Theory: Shows grounding improves answers (see theory.py #4).
# Simulated data: More facts = better accuracy
facts_used = [0, 1, 2, 3, 4, 5]
accuracy = [60, 70, 80, 85, 90, 95]  # % correct

plt.figure(figsize=(8, 5))
sns.lineplot(x=facts_used, y=accuracy, marker="o")
plt.title("Accuracy Improves with More Grounding Facts")
plt.xlabel("Number of Facts Used")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()

# 3.2 Diagram to Sketch: RAG Flow
# Theory: Visualizes how grounding works (see theory.py #4).
# Instructions for your notebook:
# - Draw a flowchart:
#   - Box 1: "Question" (e.g., "What’s France’s capital?").
#   - Arrow to Box 2: "Find Facts" (draw a library icon).
#   - Box 2 to Box 3: "Combine Question + Facts".
#   - Box 3 to Box 4: "Write Answer" (e.g., "Paris").
#   - Side Path: Dashed arrow from Question to "Wrong Answer" (no facts, show a red X).
# Analogy: Like a librarian finding a book before summarizing it.

# Reflection: Run the plot. Sketch the diagram. How does seeing this help plan experiments?
