# exercises.py: Practice Problems for Grounded NLG
# This file has exercises to build your skills, with solutions.
# Theory: Tests template and math skills (see theory.py #3, #4).
# Run with numpy: pip install numpy

import numpy as np


# Exercise 1: Template NLG
# Task: Write a function to generate a grounded sentence from a fact dictionary.
def exercise_template(fact_dict):
    return f"Did you know? {fact_dict.get('fact', 'No fact provided')}"


# Test
print(
    "Exercise 1:", exercise_template({"fact": "The sun is a star."})
)  # Output: Did you know? The sun is a star.

# Exercise 2: Cosine Similarity
# Task: Calculate cosine similarity for two vectors.
# Vectors: A=[0.8, 0.6], B=[0.9, 0.5]
# By Hand: Dot = 0.8*0.9 + 0.6*0.5 = 1.02
# Norms: sqrt(0.64 + 0.36) = 1, sqrt(0.81 + 0.25) = 1.02
# Similarity: 1.02 / (1 * 1.02) ≈ 1


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Test
a = np.array([0.8, 0.6])
b = np.array([0.9, 0.5])
print("Exercise 2:", cosine_similarity(a, b))  # Output: ~1.0

# Reflection: Try new vectors in Exercise 2. How does similarity affect grounding?
# Task: Modify exercise_template to use a different format (e.g., "Fact: [fact]").
