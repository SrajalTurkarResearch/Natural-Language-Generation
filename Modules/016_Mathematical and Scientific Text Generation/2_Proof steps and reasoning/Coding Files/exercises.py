# exercises.py
# Practical Exercises with Solutions.
# Hands-on like Tesla's experiments.

from transformers import pipeline  # For LLM

# Exercise 1: Basic CoT
generator = pipeline("text-generation", model="gpt2")
prompt = "What is 5+5? Let's think step by step."
result = generator(prompt, max_length=50)
print("Exercise 1 Solution:")
print(result[0]["generated_text"])

# Exercise 2: Build Proof Tree (adapt visualization)
# See proof_tree_visualization.py for full code; here a print-based version
print("\nExercise 2 Solution: Simple Text Tree")
print("Fact1 --> Int1")
print("Fact2 --> Int1")
print("Int1 --> Hypothesis")
# Explanation: For full graph, run proof_tree_visualization.py. This builds self-learning skills.
