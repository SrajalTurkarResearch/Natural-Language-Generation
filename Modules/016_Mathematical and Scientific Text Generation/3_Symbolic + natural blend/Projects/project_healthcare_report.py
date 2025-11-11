# project_healthcare_report.py
# Real-World: Generate Patient Reports from EMR Data
# Use: Hospitals, Telemedicine, EHR Systems

import pandas as pd
from utils import neural_summary_nlg  # Reuse from previous


def healthcare_nlg(emr_data):
    """
    Input: EMR dict with symptoms, vitals
    Output: Clinical report with urgency
    """
    fever = emr_data.get("fever", 98.6)
    bp = emr_data.get("blood_pressure", "120/80")
    symptoms = emr_data.get("symptoms", "")
    age = emr_data.get("age", 30)

    # === SYMBOLIC: Medical Logic Rules ===
    alerts = []
    if fever > 103:
        alerts.append("CRITICAL: Immediate ER referral.")
    elif fever > 100.4:
        alerts.append("URGENT: See physician within 24 hours.")

    if "chest pain" in symptoms.lower():
        alerts.append("CARDIAC: Rule out heart condition.")

    if age > 65 and fever > 100:
        alerts.append("ELDERLY: High risk. Monitor closely.")

    # === NEURAL: Fluent Clinical Narrative ===
    input_text = (
        f"Patient is {age} years old with fever {fever}°F, "
        f"blood pressure {bp}, and symptoms: {symptoms}."
    )
    narrative = neural_summary_nlg(input_text)

    # === FINAL REPORT ===
    report = f"CLINICAL SUMMARY:\n{narrative}\n"
    if alerts:
        report += "\nALERTS:\n" + "\n".join(f"• {a}" for a in alerts)
    return report


# === REAL DATA TEST ===
if __name__ == "__main__":
    emr = {
        "age": 72,
        "fever": 101.8,
        "blood_pressure": "150/90",
        "symptoms": "cough, fatigue, mild chest discomfort",
    }
    print(healthcare_nlg(emr))
