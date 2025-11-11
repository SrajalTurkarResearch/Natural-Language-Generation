# attention_sim.py: Simulating Transformer Attention Mechanism
#
# Welcome, future researcher! This script ties to Section 6 of the NL → Code tutorial,
# where we explored the attention mechanism—the heart of transformers. Attention lets
# models like Codex focus on key words (e.g., “add” in “Add two numbers”) to write code.
#
# Analogy: It’s like highlighting important notes in your textbook while ignoring fluff.
# Why this matters: Attention is why Codex and AlphaCode understand context, crucial for
# generating accurate code in research (e.g., physics simulations).
#
# This script simulates scaled dot-product attention with simple 2D vectors. Run it to
# see how the model weighs words. Copy into your notebook and ask: “Why does attention
# help my AI understand instructions?”

import numpy as np

# Define vectors for a toy example
# Imagine words: “Add” (Query), “two” and “numbers” (Keys/Values)
Q = np.array([1, 0])  # Query: “Add”
K = np.array([[1, 0], [0, 1]])  # Keys: “two”, “numbers”
V = np.array([[2, 0], [3, 1]])  # Values: info for “two”, “numbers”
d_k = 2  # Dimension for scaling

# Step 1: Calculate attention scores (dot product)
scores = np.dot(Q, K.T) / np.sqrt(d_k)  # Scale by sqrt(d_k) for stability

# Step 2: Apply softmax to get weights (sum to 1)
exp_scores = np.exp(scores)
weights = exp_scores / np.sum(exp_scores)

# Step 3: Compute output as weighted sum of values
output = np.dot(weights, V)

# Print results
print(f"Attention Scores: {scores}")
print(f"Weights: {weights}")
print(f"Output: {output}")

# Explanation: Weights [0.67, 0.33] mean 67% focus on “two”, 33% on “numbers”.
# Visual Idea: Imagine lines from “Add” to “two” (thick) and “numbers” (thin).
# Notebook Tip: Try vectors [1,1], [0,1], [2,0]. Ask: “How does attention help in my field?”
