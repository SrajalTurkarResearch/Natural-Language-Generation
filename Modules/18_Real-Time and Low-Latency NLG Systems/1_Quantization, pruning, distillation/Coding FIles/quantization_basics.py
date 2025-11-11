# quantization_basics.py
# Demonstrates basic uniform affine quantization for NLG model weights.
# Theory: Reduces bit-width to compress models, e.g., FP32 to INT8.
# Math: For tensor T, compute scale s and zero-point z to map to discrete values.
# Error analysis: MSE <= s^2 / 12 under uniform noise.
# As a researcher, experiment with different bit-widths to observe trade-offs in NLG accuracy.

import numpy as np

# Example weights from a hypothetical NLG model layer
weights = np.array([1.2, 3.7, -0.5])

# Bit-width (e.g., 8 for INT8)
b = 8

# Compute min and max of the tensor
min_val, max_val = weights.min(), weights.max()

# Scale factor: Spread the range over 2^b - 1 levels
s = (max_val - min_val) / (2**b - 1)

# Zero-point: Offset for asymmetric quantization
z = np.round(-min_val / s)

# Quantize: Round and clip to [0, 2^b - 1]
q = np.clip(np.round(weights / s + z), 0, 2**b - 1)

# Dequantize: Approximate original values
deq = s * (q - z)

# Output results
print("Original weights:", weights)
print("Quantized values:", q)
print("Dequantized approximations:", deq)
print("Quantization error (MSE):", np.mean((weights - deq) ** 2))

# Research extension: Apply to a full matrix and measure impact on matrix multiplication in transformers.
