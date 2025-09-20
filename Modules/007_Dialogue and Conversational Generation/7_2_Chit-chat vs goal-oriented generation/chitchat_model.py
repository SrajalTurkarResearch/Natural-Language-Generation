# Chit-Chat Transformer Model and Visualization for NLG
# Implements a transformer for open-domain conversation

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Define Chit-Chat Transformer
class ChitChatTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=4)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        return self.fc(out)


# Instantiate and Test
model = ChitChatTransformer(vocab_size=10000)
print("Chit-Chat Transformer Model:", model)

# Visualize Metrics
metrics = ["Engagement", "Accuracy", "Structure"]
chit_scores = [0.9, 0.6, 0.4]
x = np.arange(len(metrics))
plt.bar(x, chit_scores, 0.4, label="Chit-Chat", color="blue")
plt.xticks(x, metrics)
plt.ylabel("Normalized Score")
plt.title("Chit-Chat NLG Metrics (2025)")
plt.legend()
plt.show()
