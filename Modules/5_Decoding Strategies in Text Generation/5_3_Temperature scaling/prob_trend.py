# prob_trend.py: Analyze Top Probability Trend Across Temperatures
# Theory: As T increases, the top word’s probability decreases (flatter distribution).
# - Useful for tuning NLG models to balance creativity and reliability.
# Real-World: In case_study3.md, Google used T=0.2 for translation accuracy.
# Mini-Project: Study how T affects top word selection.

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def softmax_with_temp(logits, temperature=1.0):
    scaled = logits / temperature
    return F.softmax(torch.tensor(scaled), dim=0).numpy()


# Data setup
logits = np.array([4.0, 2.0, 1.0, 0.0])
temps = np.linspace(0.1, 3.0, 100)
probs_top = [softmax_with_temp(logits, t)[0] for t in temps]

# Plot top probability trend
plt.plot(temps, probs_top)
plt.xlabel("Temperature")
plt.ylabel("Probability of Top Word (mat)")
plt.title("Effect of Temperature on Top Probability")
plt.grid(True)
plt.show()

# Researcher Tip: Log the T where prob drops below 0.5 for your model.
# Extend: Test with real model logits (e.g., from Hugging Face Transformers).
