# reward_feedback.py
# Purpose: Implement reward feedback in NLG using PyTorch, simulating RL
# Inspired by Hinton’s neural networks and Sutton’s reinforcement learning
# Use: Run to train a simple model with a reward based on text length

import torch
import torch.nn as nn


# Define a simple neural model for NLG (placeholder for text generation)
class SimpleNLG(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer: Dummy input (10 features) to output probability
        self.fc = nn.Linear(10, 1)  # Simplified for demo

    def forward(self, x):
        # Output probability (0 to 1)
        return torch.sigmoid(self.fc(x))


# Define a reward function (example: based on text length)
def reward(text):
    # Simple reward: Normalize length to 0-1 scale (max 100)
    return len(text) / 100.0


# Initialize model and optimizer
model = SimpleNLG()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Simulate input (random vector, representing encoded text)
input = torch.rand(10)

# Forward pass: Generate probability
pred = model(input)

# Calculate reward for a sample text
sample_text = "Sample text for testing"
r = reward(sample_text)

# Compute loss (REINFORCE-style: maximize reward * log(prob))
loss = -r * torch.log(pred)

# Backpropagate and update model
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Output result
print(f"Trained with reward: {r}")

# Explanation for researchers:
# - This simulates a neural NLG model learning via rewards
# - Reward function is simple (length-based); real systems use coherence, accuracy
# - Try modifying reward (e.g., +1 for keyword inclusion) and rerun
# - Next step: Combine with constraints for neurosymbolic approach
