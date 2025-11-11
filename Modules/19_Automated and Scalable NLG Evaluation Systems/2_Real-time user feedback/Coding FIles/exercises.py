# Exercises for Self-Learning
# This script includes functions from the exercises in the notebook.

import numpy as np


# Exercise 1: Cosine Similarity for Word Embeddings
def cosine_sim(a, b):
    # Compute dot product divided by norms
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0


# Example vectors
vec_a = np.array([1, 2])
vec_b = np.array([2, 3])
print("Cosine Similarity:", cosine_sim(vec_a, vec_b))


# Exercise 2: Extend Feedback Function (Stub for Multi-Iterations)
def multi_iter_feedback(prompt, feedbacks):
    current_prompt = prompt
    for fb in feedbacks:
        current_prompt += f" {fb}"
        # Simulate generation (replace with actual model call)
        print(f"Generated after '{fb}': {current_prompt}")


# Example
multi_iter_feedback("Write a story.", ["Make it short.", "Add humor."])

# Exercise 3: Plot Perplexity (Simulated)
import matplotlib.pyplot as plt

steps = [1, 2, 3, 4, 5]
perplexity = [10, 7, 5, 4, 3]  # Simulated decreasing perplexity
plt.plot(steps, perplexity)
plt.title("Simulated Perplexity vs. Training Steps")
plt.xlabel("Steps")
plt.ylabel("Perplexity")
plt.show()
