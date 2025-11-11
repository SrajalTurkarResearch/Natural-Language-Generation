# visualization_reward.py
# Purpose: Visualize reward feedback learning curve using Matplotlib
# Inspired by da Vinci’s sketches and scientific plotting
# Use: Run to plot a simulated reward improvement over iterations

import matplotlib.pyplot as plt
import numpy as np

# Simulate training iterations and rewards
iterations = np.arange(100)
rewards = np.log(iterations + 1)  # Logarithmic growth (mimics learning)

# Create plot
plt.plot(iterations, rewards)
plt.xlabel("Training Iterations")
plt.ylabel("Reward Score")
plt.title("Reward Feedback Learning Curve")
plt.grid(True)
plt.show()

# Explanation for researchers:
# - Visualizes how rewards improve over training, like a scientist tracking progress
# - Logarithmic curve simulates diminishing returns in learning
# - Try changing rewards (e.g., np.exp(iterations/50)) for different curves
# - Next step: Plot real rewards from reward_feedback.py
