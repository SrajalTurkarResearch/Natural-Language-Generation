# distillation_visualization.py
# Plots teacher vs. student probabilities for distillation analysis.
# Theory: Soft labels reveal uncertainties not in hard labels.

import numpy as np
import matplotlib.pyplot as plt

# Example probabilities
probs_t = [0.506, 0.307, 0.187]
probs_s = [0.475, 0.289, 0.236]
labels = ["Token1", "Token2", "Token3"]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width / 2, probs_t, width, label="Teacher")
ax.bar(x + width / 2, probs_s, width, label="Student")
ax.set_ylabel("Probability")
ax.set_title("Teacher vs. Student Token Probabilities in NLG")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Research note: Use for debugging distillation in sequence generation.
