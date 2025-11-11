# distillation_basics.py
# Computes KL-divergence loss for knowledge distillation in NLG.
# Theory: Student mimics teacher's soft probabilities.
# Math: p = softmax(z / tau); loss scaled by tau^2.
# In NLG, aids small models in generating nuanced text.

import torch
import torch.nn.functional as F

# Example teacher and student logits (pre-softmax outputs)
z_t = torch.tensor([2.0, 1.0, 0.0])
z_s = torch.tensor([1.5, 1.0, 0.5])

# Temperature for softening
tau = 2.0

# Soft probabilities
p_t = F.softmax(z_t / tau, dim=0)
p_s = F.softmax(z_s / tau, dim=0)

# KD loss (KL-divergence, scaled)
kl = F.kl_div(p_s.log(), p_t, reduction="batchmean") * tau**2

print("Teacher probs:", p_t.numpy())
print("Student probs:", p_s.numpy())
print("KL Loss:", kl.item())

# Research extension: Combine with cross-entropy for full training.
