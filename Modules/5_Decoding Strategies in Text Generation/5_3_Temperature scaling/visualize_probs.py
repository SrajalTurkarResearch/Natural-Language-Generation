# visualize_probs.py: Visualize Temperature Scaling Effects
# Theory: Temperature changes the shape of probability distributions.
# - Low T: Tall, narrow bars (deterministic, like a focused laser).
# - High T: Flat bars (random, like a scattered light beam).
# Visualization helps understand trade-offs in NLG (accuracy vs. creativity).
# Real-World: Used in debugging NLG models (e.g., GPT tuning in case_study1.md).

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def softmax_with_temp(logits, temperature=1.0):
    scaled = logits / temperature
    return F.softmax(torch.tensor(scaled), dim=0).numpy()


# Data setup
logits = np.array([4.0, 2.0, 1.0, 0.0])
words = ["mat", "roof", "chair", "moon"]
temps = [0.5, 1.0, 2.0]

# Plot probabilities for each temperature
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, t in enumerate(temps):
    probs = softmax_with_temp(logits, t)
    axs[i].bar(words, probs)
    axs[i].set_title(f"Temperature = {t}")
    axs[i].set_ylim(0, 1)
    axs[i].set_ylabel("Probability")
plt.tight_layout()
plt.show()

# Researcher Tip: Save this plot as 'probs.png' for your notes.
# Experiment: Change logits or add more temps (e.g., 0.1, 3.0).
