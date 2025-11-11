# visualize_metrics.py
# Visualization for EaaS Results

import matplotlib.pyplot as plt
import numpy as np


def radar_chart(models_data, labels):
    """
    models_data: list of dicts with metric names as keys
    labels: list of metric names
    """
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["blue", "green", "red", "purple"]
    for idx, (name, scores) in enumerate(models_data.items()):
        values = [scores.get(label, 0) for label in labels]
        values += values[:1]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=name,
            color=colors[idx % len(colors)],
        )
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("NLG Model Comparison (EaaS Radar)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.show()


# === TEST ===
if __name__ == "__main__":
    models = {
        "GPT-3": {
            "Accuracy": 0.88,
            "Speed": 0.65,
            "Fairness": 0.70,
            "Robustness": 0.75,
        },
        "T5": {"Accuracy": 0.82, "Speed": 0.90, "Fairness": 0.85, "Robustness": 0.68},
        "LLaMA": {
            "Accuracy": 0.91,
            "Speed": 0.55,
            "Fairness": 0.78,
            "Robustness": 0.80,
        },
    }
    labels = ["Accuracy", "Speed", "Fairness", "Robustness"]
    radar_chart(models, labels)
