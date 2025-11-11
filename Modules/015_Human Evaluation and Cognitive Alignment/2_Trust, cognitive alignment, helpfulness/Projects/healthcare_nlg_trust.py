# healthcare_nlg_trust.py
# --------------------------------------------------------------
# REAL-WORLD PROJECT: Generate patient reports + evaluate TRUST
# Use case: Radiology AI assistant (like IBM Watson Health)
# --------------------------------------------------------------

# Install once: pip install transformers torch pandas matplotlib scikit-learn

import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# --------------------------------------------------------------
# 1. Simulate real medical data (or load from CSV)
# --------------------------------------------------------------
data = [
    {
        "id": 1,
        "finding": "lung opacity",
        "location": "left lower lobe",
        "severity": "moderate",
        "doctor_note": "Likely pneumonia. Recommend antibiotics.",
    },
    {
        "id": 2,
        "finding": "heart enlargement",
        "location": "cardiomegaly",
        "severity": "mild",
        "doctor_note": "Monitor blood pressure.",
    },
    {
        "id": 3,
        "finding": "no abnormality",
        "location": "",
        "severity": "none",
        "doctor_note": "Patient is healthy.",
    },
]

df = pd.DataFrame(data)

# --------------------------------------------------------------
# 2. NLG: Generate patient-friendly report
# --------------------------------------------------------------
generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")


def generate_report(row):
    prompt = f"Patient has {row['finding']} in {row['location']}, severity {row['severity']}. Explain simply:"
    output = generator(prompt, max_length=80, num_return_sequences=1, temperature=0.7)[
        0
    ]["generated_text"]
    return output.split("Explain simply:")[-1].strip()


df["ai_report"] = df.apply(generate_report, axis=1)

# --------------------------------------------------------------
# 3. TRUST EVALUATION: Compare AI vs. Doctor (ground truth)
# --------------------------------------------------------------
# Simple keyword match for accuracy (real systems use BERTScore/fact-check)
keywords = {
    "lung opacity": ["lung", "opacity", "shadow"],
    "heart enlargement": ["heart", "enlarged", "cardiomegaly"],
    "no abnormality": ["normal", "healthy", "clear"],
}


def check_accuracy(ai_text, finding):
    key = (
        finding.split()[0] + " " + finding.split()[1]
        if len(finding.split()) > 1
        else finding
    )
    if key not in keywords:
        return 0.0
    return 1.0 if any(word in ai_text.lower() for word in keywords[key]) else 0.0


df["accuracy"] = df.apply(
    lambda x: check_accuracy(x["ai_report"], x["finding"]), axis=1
)
df["transparency"] = 0.8  # Assume model shows source
df["reliability"] = 0.9  # Consistent over runs
df["trust_score"] = (df["accuracy"] + df["transparency"] + df["reliability"]) / 3

# --------------------------------------------------------------
# 4. VISUALIZATION: Trust per case
# --------------------------------------------------------------
plt.figure(figsize=(10, 5))
bars = plt.bar(
    df["id"].astype(str), df["trust_score"], color=["#66c2a5", "#fc8d62", "#8da0cb"]
)
plt.title("Trust Score in AI-Generated Medical Reports", fontsize=14)
plt.ylabel("Trust Score (0–1)")
plt.xlabel("Patient Case ID")
for i, bar in enumerate(bars):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f'{df.iloc[i]["trust_score"]:.2f}',
        ha="center",
        fontsize=10,
    )
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------------------
df[["id", "finding", "doctor_note", "ai_report", "trust_score"]].to_csv(
    "medical_nlg_trust_results.csv", index=False
)
print(df[["id", "ai_report", "trust_score"]])
print("\nProject complete: healthcare_nlg_trust.py")
