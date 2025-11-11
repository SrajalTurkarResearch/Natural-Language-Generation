# cognitive_alignment.py
# --------------------------------------------------------------
# Cognitive Alignment – vector similarity between human & AI
# --------------------------------------------------------------

import numpy as np
from scipy.spatial.distance import cosine, jensenshannon
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. Dummy embeddings (3-dim concept vectors)
# --------------------------------------------------------------
human_vec = np.array([0.5, 0.3, 0.8])  # e.g. [fruit, red, healthy] for "apple"
ai_vec = np.array([0.4, 0.2, 0.9])

# Normalise for cosine
human_norm = human_vec / np.linalg.norm(human_vec)
ai_norm = ai_vec / np.linalg.norm(ai_vec)

cos_sim = 1 - cosine(human_norm, ai_norm)
print(f"Cosine similarity = {cos_sim:.4f}")

# --------------------------------------------------------------
# 2. Jensen-Shannon Divergence on probability distributions
# --------------------------------------------------------------
p = np.array([0.6, 0.4])  # human preference over two options
q = np.array([0.5, 0.5])  # AI preference
jsd = jensenshannon(p, q) ** 2
print(f"JSD (squared) = {jsd:.5f}")

# --------------------------------------------------------------
# 3. Visualise the two vectors
# --------------------------------------------------------------
fig, ax = plt.subplots()
origin = np.zeros(2)

ax.quiver(*origin, human_vec[0], human_vec[1], color="b", scale=3, label="Human")
ax.quiver(*origin, ai_vec[0], ai_vec[1], color="r", scale=3, label="AI")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_title("Human vs. AI concept vectors")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
