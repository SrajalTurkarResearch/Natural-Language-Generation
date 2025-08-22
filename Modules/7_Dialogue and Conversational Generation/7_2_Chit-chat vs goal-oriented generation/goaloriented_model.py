# Goal-Oriented Intent Classifier and Visualization for NLG
# Implements a classifier for task-oriented dialogues

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Define Goal-Oriented Classifier
class GoalClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


# Instantiate and Test
model = GoalClassifier(input_size=768, num_classes=10)  # e.g., BERT embeddings
print("Goal-Oriented Classifier Model:", model)

# Visualize Metrics
metrics = ["Engagement", "Accuracy", "Structure"]
goal_scores = [0.5, 0.9, 0.9]
x = np.arange(len(metrics))
plt.bar(x, goal_scores, 0.4, label="Goal-Oriented", color="red")
plt.xticks(x, metrics)
plt.ylabel("Normalized Score")
plt.title("Goal-Oriented NLG Metrics (2025)")
plt.legend()
plt.show()
