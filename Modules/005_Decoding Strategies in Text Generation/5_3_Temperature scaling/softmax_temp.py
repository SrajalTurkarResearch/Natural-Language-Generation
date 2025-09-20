# softmax_temp.py: Implementation of Softmax with Temperature Scaling
# Theory: Temperature scaling adjusts probabilities in NLG to control output diversity.
# - Low T (e.g., 0.5): Sharp probs, predictable outputs (like a frozen lake).
# - High T (e.g., 2.0): Flat probs, creative outputs (like boiling water).
# - Formula: p_i = exp(z_i / T) / sum(exp(z_j / T))
# Real-World: Used in chatbots (low T for facts) or story generators (high T).
# Case Study Reference: case_study1.md (GPT-2 used T=1.2 to reduce repetition).

import numpy as np
import torch
import torch.nn.functional as F


def softmax_with_temp(logits, temperature=1.0):
    """
    Compute softmax with temperature scaling.
    Args:
        logits (np.array): Raw scores for each word.
        temperature (float): Scaling factor for logits.
    Returns:
        np.array: Probabilities after softmax.
    """
    scaled = logits / temperature
    return F.softmax(torch.tensor(scaled), dim=0).numpy()


# Example: Predict next word after "The cat sat on the..."
logits = np.array([4.0, 2.0, 1.0, 0.0])  # Scores for ['mat', 'roof', 'chair', 'moon']
words = ["mat", "roof", "chair", "moon"]
temps = [0.5, 1.0, 2.0]

# Compute and print probabilities for different temperatures
for t in temps:
    probs = softmax_with_temp(logits, t)
    print(f"Temperature {t}:")
    for word, prob in zip(words, probs):
        print(f"  {word}: {prob:.3f}")

# Researcher Tip: Try T=1.5 and observe probability flattening.
# Save output to analyze in your notes for patterns.
