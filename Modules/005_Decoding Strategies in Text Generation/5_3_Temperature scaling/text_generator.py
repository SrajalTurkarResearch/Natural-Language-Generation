# text_generator.py: Simple Text Generation with Temperature
# Theory: Temperature controls creativity in text generation.
# - Low T: Predictable sequences (e.g., "The cat sat on the mat").
# - High T: Varied outputs (e.g., "The cat sat on the moon").
# Real-World: Chatbots (case_study2.md: BioGPT uses T=0.6 for medical reports).
# Mini-Project: Generate text and analyze output diversity.

import numpy as np
import torch
import torch.nn.functional as F


def softmax_with_temp(logits, temperature=1.0):
    scaled = logits / temperature
    return F.softmax(torch.tensor(scaled), dim=0).numpy()


# Simulated language model vocabulary and logits
vocab = ["the", "cat", "sat", "on", "mat", "roof"]
logit_dict = {"the cat sat on the": [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]}  # Simplified


def generate_text(prompt, temp=1.0, length=5):
    """
    Generate text by sampling words based on temperature-scaled probs.
    Args:
        prompt (str): Starting text.
        temp (float): Temperature for scaling.
        length (int): Number of words to generate.
    Returns:
        str: Generated text.
    """
    text = prompt
    for _ in range(length):
        logits = np.array(logit_dict.get(text, [1 / len(vocab)] * len(vocab)))
        probs = softmax_with_temp(logits, temp)
        next_word = np.random.choice(vocab, p=probs)
        text += " " + next_word
    return text


# Test generation
print("Low T (0.5):", generate_text("the cat sat on the", temp=0.5))
print("High T (2.0):", generate_text("the cat sat on the", temp=2.0))

# Researcher Tip: Run 10 times for each T, log outputs, and compute diversity (unique words).
# Extend: Add more prompts or vocab for a major project.
