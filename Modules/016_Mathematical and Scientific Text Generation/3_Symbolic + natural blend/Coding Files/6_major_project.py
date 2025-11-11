# 6_major_project.py
# Major Project: Medical Patient Report Generator

from utils import neural_summary_nlg


def medical_nlg(patient):
    """
    Generate medical report with symbolic alerts + neural fluency.
    """
    fever = patient["fever"]
    symptom = patient["symptom"]

    # === SYMBOLIC: Medical rules ===
    if fever > 103:
        urgency = "CRITICAL: Go to ER immediately."
    elif fever > 100:
        urgency = "Urgent: Consult doctor today."
    else:
        urgency = "Monitor symptoms."

    # === NEURAL: Fluent description ===
    desc = neural_summary_nlg(f"Patient has fever of {fever}°F and {symptom}.")

    return f"{desc} {urgency}"


# === TEST with synthetic data ===
if __name__ == "__main__":
    patients = [
        {"fever": 102.5, "symptom": "cough and fatigue"},
        {"fever": 104.0, "symptom": "severe headache"},
        {"fever": 98.6, "symptom": "mild sore throat"},
    ]

    print("MEDICAL REPORTS\n" + "=" * 50)
    for p in patients:
        print(medical_nlg(p))
        print("-" * 40)
