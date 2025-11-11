# finance_risk_explainer.py
"""
Real-World Project: Finance Formula Explainer
Use Case: Explain Value at Risk (VaR), option pricing to non-experts.
"""

from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

explainer = pipeline("text2text-generation", model="google/flan-t5-large")


def explain_var():
    prompt = "Explain Value at Risk (VaR) to a business manager in simple terms."
    return explainer(prompt, max_length=120)[0]["generated_text"]


def plot_var_distribution():
    np.random.seed(42)
    returns = np.random.normal(-0.001, 0.02, 1000)
    var_95 = np.percentile(returns, 5)

    plt.figure(figsize=(8, 5))
    plt.hist(returns, bins=50, alpha=0.7, color="skyblue")
    plt.axvline(var_95, color="red", linestyle="--", label=f"95% VaR = {var_95:.3%}")
    plt.title("Daily Return Distribution with 95% VaR")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Finance Risk Explainer\n")
    print(explain_var())
    print("\nGenerating VaR visualization...")
    plot_var_distribution()
