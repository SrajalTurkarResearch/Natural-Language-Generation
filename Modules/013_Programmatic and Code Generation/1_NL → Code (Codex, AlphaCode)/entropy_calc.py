# entropy_calc.py: Calculating Shannon Entropy for NLP
#
# Hey, aspiring scientist! This script connects to Section 1.2 of the NL → Code tutorial,
# where we learned how Claude Shannon’s information theory helps NLP measure language
# unpredictability. Entropy quantifies how “surprising” a word is, which is key for
# predicting text or code in NLG. Think of it like measuring chaos in a lab experiment!
#
# Why this matters: Low entropy means predictable text, which helps models like Codex
# generate code. This script calculates entropy for a biased coin (like a simple language
# model) to make the concept concrete.
#
# Real-World Use: In bioinformatics, entropy helps analyze DNA sequence patterns.
#
# Run this in Python to see the result. Copy into your lab notebook and ask:
# “How does entropy relate to code generation?”

import math

# Define probabilities for a biased coin (like words in a sentence)
p_heads = 0.8  # 80% chance of heads
p_tails = 0.2  # 20% chance of tails

# Shannon Entropy: H = -Σ p(x) log₂ p(x)
# This measures unpredictability (in bits)
entropy = -(p_heads * math.log2(p_heads) + p_tails * math.log2(p_tails))

# Print result
print(f"Entropy of biased coin: {entropy:.2f} bits")

# Explanation: 0.72 bits means the coin is fairly predictable (low entropy).
# Try changing p_heads to 0.5 for a fair coin (higher entropy, ~1 bit).
# Notebook Tip: Ask: “How could I use entropy to study patterns in my research data?”
