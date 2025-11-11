# trust_score.py
# --------------------------------------------------------------
# Trust modelling – simple weighted score + regression demo
# --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# 1. Trust score function (0–1 scale)
# --------------------------------------------------------------
def trust_score(accuracy: float, transparency: float, reliability: float) -> float:
    """Average of three trust pillars."""
    return (accuracy + transparency + reliability) / 3.0


# Example values (you can change them)
acc, trans, rel = 0.85, 0.70, 0.92
score = trust_score(acc, trans, rel)
print(f"Trust score = {score:.3f}")

# --------------------------------------------------------------
# 2. Linear regression toy model (optional)
# --------------------------------------------------------------
# Simulate 30 systems with random pillar values
np.random.seed(42)
n = 30
accuracy = np.random.beta(5, 2, n)
transparency = np.random.beta(4, 3, n)
reliability = np.random.beta(6, 2, n)

# Ground-truth trust (with small noise)
true_trust = (
    0.2
    + 0.3 * accuracy
    + 0.4 * transparency
    + 0.1 * reliability
    + np.random.normal(0, 0.03, n)
)

# Fit a simple OLS
X = np.column_stack([np.ones(n), accuracy, transparency, reliability])
beta = np.linalg.lstsq(X, true_trust, rcond=None)[0]
print(
    f"\nLearned coefficients -> β0:{beta[0]:.3f}, β_acc:{beta[1]:.3f}, β_trans:{beta[2]:.3f}, β_rel:{beta[3]:.3f}"
)

# --------------------------------------------------------------
# 3. Plot the three pillars vs. computed trust
# --------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(accuracy, true_trust, label="Accuracy", alpha=0.7)
plt.scatter(transparency, true_trust, label="Transparency", alpha=0.7)
plt.scatter(reliability, true_trust, label="Reliability", alpha=0.7)
plt.xlabel("Pillar value")
plt.ylabel("Observed trust")
plt.title("Trust vs. individual pillars")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
