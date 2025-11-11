# exercise_1_4bit_quantization.py
# Exercise solution: Implement 4-bit quantization.
# Theory: For b=4, 16 levels; useful for ultra-efficient NLG.
# Math: s = (max - min) / (2^4 - 1) = range / 15.

import numpy as np


def quantize_4bit(weights):
    # Compute range
    min_val, max_val = np.min(weights), np.max(weights)

    # Scale for 4 bits (15 levels)
    s = (max_val - min_val) / 15

    # Quantize (assuming min=0 for simplicity; adjust for affine)
    q = np.clip(np.round((weights - min_val) / s), 0, 15)

    return q


# Example weights
weights = np.array([0.0, 0.5, 1.0])

# Run quantization
quantized = quantize_4bit(weights)
print("Original weights:", weights)
print("4-bit quantized:", quantized)

# Research extension: Dequantize and compute error; test on real embeddings.
