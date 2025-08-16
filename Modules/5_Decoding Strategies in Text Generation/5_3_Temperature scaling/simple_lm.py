# simple_lm.py: Simulated Language Model with Temperature Scaling
# Theory: Language models predict next words using softmax with temperature.
# - Major project stub for integrating with real models (e.g., GPT-2).
# Real-World: Case_study2.md (BioGPT) shows T=0.6 for medical NLG.
# Major Project: Extend to real datasets and models (e.g., Hugging Face).

import numpy as np
import torch
import torch.nn.functional as F


class SimpleLM:
    def __init__(self, vocab):
        self.vocab = vocab
        self.logit_dict = {"the cat sat on the": [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]}

    def forward(self, prompt):
        """Simulate logits for a prompt."""
        return np.array(
            self.logit_dict.get(prompt, [1 / len(self.vocab)] * len(self.vocab))
        )


def softmax_with_temp(logits, temperature=1.0):
    scaled = logits / temperature
    return F.softmax(torch.tensor(scaled), dim=0).numpy()


def generate_text(model, prompt, temp=1.0, length=5):
    text = prompt
    for _ in range(length):
        logits = model.forward(text)
        probs = softmax_with_temp(logits, temp)
        next_word = np.random.choice(model.vocab, p=probs)
        text += " " + next_word
    return text


# Initialize and test
vocab = ["the", "cat", "sat", "on", "mat", "roof"]
model = SimpleLM(vocab)
print("Low T (0.5):", generate_text(model, "the cat sat on the", temp=0.5))
print("High T (2.0):", generate_text(model, "the cat sat on the", temp=2.0))

# Researcher Tip: Replace logit_dict with real model logits (e.g., GPT-2).
# Next Step: Integrate with Hugging Face Transformers for a full pipeline.
