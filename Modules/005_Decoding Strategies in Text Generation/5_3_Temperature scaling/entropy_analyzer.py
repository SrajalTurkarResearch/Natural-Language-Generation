# entropy_analyzer.py: Compute and Visualize Entropy in NLG
# Theory: Entropy measures randomness in probability distributions (-sum p_i log p_i).
# - Low T: Low entropy (predictable, like a tidy lab).
# - High T: High entropy (diverse, like a chaotic experiment).
# Real-World: Used in case_study2.md (BioGPT) to balance creativity and accuracy.
# Major Project: Analyze entropy trends for model tuning.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import torch
import torch.nn.functional as F


def softmax_with_temp(logits, temperature=1.0):
    scaled = logits / temperature
    return F.softmax(torch.tensor(scaled), dim=0).numpy()


def compute_entropy(logits, t):
    """
    Compute Shannon entropy of softmax probabilities.
    Args:
        logits (np.array): Raw scores.
        t (float): Temperature.
    Returns:
        float: Entropy in nats (using natural log).
    """
    probs = softmax_with_temp(logits, t)
    return entropy(probs, base=np.e)


# Data setup
logits = np.array([4.0, 2.0, 1.0, 0.0])
temps = np.linspace(0.1, 3.0, 100)
entropies = [compute_entropy(logits, t) for t in temps]

# Visualize entropy vs. temperature
plt.plot(temps, entropies)
plt.xlabel("Temperature")
plt.ylabel("Entropy (nats)")
plt.title("Entropy vs. Temperature in NLG")
plt.grid(True)
plt.show()

# Print entropy for key temperatures
for t in [0.5, 1.0, 2.0]:
    ent = compute_entropy(logits, t)
    print(f"Entropy at T={t}: {ent:.3f}")

# Researcher Tip: High entropy at T>2 suggests over-randomization.
# Experiment: Compute entropy for real LM outputs (e.g., GPT-2 in case_study1.md).
