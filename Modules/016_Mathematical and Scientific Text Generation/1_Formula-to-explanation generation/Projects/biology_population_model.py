# biology_population_model.py
"""
Real-World Project: Biology & Epidemic Model Explainer
Use Case: Explain SIR model or logistic growth to public health officials.
"""

from transformers import pipeline
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

explainer = pipeline("text2text-generation", model="t5-base")


def logistic_growth_explanation():
    prompt = "Explain the logistic growth model P = K / (1 + (K/P0 - 1) e^(-r t)) to a government official."
    return explainer(prompt, max_length=150)[0]["generated_text"]


def plot_logistic(P0=10, K=100, r=0.1, t_max=50):
    t = np.linspace(0, t_max, 200)
    P = K / (1 + (K / P0 - 1) * np.exp(-r * t))

    plt.figure(figsize=(9, 5))
    plt.plot(t, P, label="Population", color="green")
    plt.axhline(K, color="red", linestyle="--", label=f"Carrying Capacity K={K}")
    plt.title("Logistic Population Growth")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Biology Population Model Explainer\n")
    print(logistic_growth_explanation())
    print("\nSimulating logistic growth...")
    plot_logistic()
