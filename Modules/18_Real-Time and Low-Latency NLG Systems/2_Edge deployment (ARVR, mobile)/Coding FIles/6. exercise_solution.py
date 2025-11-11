# exercise_solution.py
# Purpose: Solutions to exercises for self-learning.
# Exercise 1: Pruning - Set small weights to zero.
# Theory: Structured pruning removes channels; here unstructured for simplicity.
import torch

weight = torch.tensor([0.05, 0.005, 0.1])  # Sample weights
threshold = 0.01
pruned = weight.clone()
pruned[abs(pruned) < threshold] = 0  # Apply mask
print("Pruned Weights:", pruned)  # Output: [0.05, 0.0, 0.1]

# Exercise 2: Quantization error - MSE
original = torch.tensor([1.2, 3.4, 5.6])
# Simulate quant: Assume min=0, max=6, bits=8, s=6/255≈0.0235, z=0
quantized = torch.round(original / 0.0235) * 0.0235  # Approx
mse = torch.mean((original - quantized) ** 2)
print("MSE Error:", mse.item())  # Low error indicates good quantization

# As scientist: Extend to full model; compare pre/post-prune accuracy
