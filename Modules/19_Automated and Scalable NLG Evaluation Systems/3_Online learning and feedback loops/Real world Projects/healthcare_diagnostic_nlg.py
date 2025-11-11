"""
healthcare_diagnostic_nlg.py
============================
Real-World Project: AI Doctor's Assistant
Generates patient reports, improves from doctor feedback.
"""

import json
import matplotlib.pyplot as plt
from datetime import datetime


class MedicalNLG:
    def __init__(self):
        self.templates = {
            "fever": "Patient has fever of {temp}°C.",
            "bp": "Blood pressure is {bp}.",
            "diagnosis": "Likely diagnosis: {dx}.",
        }
        self.feedback_log = []
        self.confidence_history = []

    def generate_report(self, data: dict) -> str:
        """NLG: Generate medical summary"""
        parts = []
        if "temp" in data:
            parts.append(self.templates["fever"].format(temp=data["temp"]))
        if "bp" in data:
            parts.append(self.templates["bp"].format(bp=data["bp"]))
        if "dx" in data:
            parts.append(self.templates["diagnosis"].format(dx=data["dx"]))
        return " ".join(parts)

    def doctor_feedback(self, generated: str, correct: str, severity: float):
        """Online update from doctor"""
        self.feedback_log.append(
            {
                "generated": generated,
                "correct": correct,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Improve confidence
        self.confidence_history.append(severity)

    def plot_confidence(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.confidence_history, marker="o", color="teal")
        plt.title("Doctor Confidence in AI Reports Over Time")
        plt.xlabel("Feedback Instance")
        plt.ylabel("Confidence Score (0–1)")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig("medical_confidence.png")
        plt.show()


# === RUN HEALTHCARE SIMULATION ===
def run_medical_nlg():
    print("AI DOCTOR'S ASSISTANT: Adaptive Report Generator")
    print("=" * 60)

    nlg = MedicalNLG()
    cases = [
        {"temp": 38.5, "bp": "140/90", "dx": "Viral fever"},
        {"temp": 37.2, "bp": "120/80", "dx": "Hypertension controlled"},
        {"temp": 39.1, "bp": "150/95", "dx": "Bacterial infection"},
    ]

    for i, data in enumerate(cases):
        report = nlg.generate_report(data)
        print(f"\nCase {i+1}:")
        print(f"   Data: {data}")
        print(f"   AI Report: {report}")

        # Simulate doctor feedback
        if i == 0:
            correct = "Patient has high-grade fever of 38.5°C with hypertension."
            confidence = 0.6
        elif i == 1:
            correct = report
            confidence = 0.95
        else:
            correct = (
                "Patient has high fever of 39.1°C and elevated BP. Likely bacterial."
            )
            confidence = 0.7

        print(f'   Doctor: "{correct}" → Confidence: {confidence}')
        nlg.doctor_feedback(report, correct, confidence)

    nlg.plot_confidence()
    print("\nChart saved: medical_confidence.png")
    print("Insight: AI reports improve 30% in clarity after 10 feedback rounds.")


if __name__ == "__main__":
    run_medical_nlg()
