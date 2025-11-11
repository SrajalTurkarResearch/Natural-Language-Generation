# medical_report_nlg.py
# Real-World NLG: Doctor-Readable Patient Summary
# Used in: Epic Systems, Cerner, NHS


def generate_patient_summary(patient_data):
    """
    Convert EHR JSON to human-readable summary
    """
    name = patient_data["name"]
    age = patient_data["age"]
    gender = patient_data["gender"]
    vitals = patient_data["vitals"]
    diagnosis = patient_data["diagnosis"]
    meds = patient_data["medications"]

    # Risk assessment
    risk = "low"
    if vitals["bp"][0] > 140 or vitals["hr"] > 100:
        risk = "elevated"
    if any(m["critical"] for m in meds):
        risk = "high"

    # Generate
    summary = f"""
PATIENT SUMMARY - {name.upper()}
Age: {age} | Gender: {gender} | Risk Level: {risk.upper()}

VITALS:
- Blood Pressure: {vitals['bp'][0]}/{vitals['bp'][1]} mmHg
- Heart Rate: {vitals['hr']} bpm
- Temperature: {vitals['temp']}°C
- Oxygen: {vitals['o2']}% 

DIAGNOSIS:
{diagnosis}

CURRENT MEDICATIONS:
"""
    for med in meds:
        urgency = " (URGENT)" if med.get("critical") else ""
        summary += f"- {med['name']} {med['dose']} {urgency}\n"

    summary += f"\nNEXT STEPS: {'Immediate review required.' if risk == 'high' else 'Routine follow-up.'}"
    return summary


# === SAMPLE EHR DATA ===
patient = {
    "name": "Amit Sharma",
    "age": 58,
    "gender": "Male",
    "vitals": {"bp": [155, 95], "hr": 110, "temp": 37.8, "o2": 94},
    "diagnosis": "Type 2 Diabetes with Hypertension",
    "medications": [
        {"name": "Metformin", "dose": "500mg BID"},
        {"name": "Lisinopril", "dose": "10mg daily", "critical": True},
    ],
}

if __name__ == "__main__":
    print(generate_patient_summary(patient))
