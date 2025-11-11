# medical_application.py
# Neurosymbolic NLG for medical reports.
# Author: Applying to practical domains, per engineering principles.
# Requirements: transformers, torch.
# Usage: Run with patient data.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def medical_nlg(patient_data):
    """
    Generate medical report: Rule base + neural diagnosis extension.
    Args:
        patient_data (dict): Patient info.
    Returns:
        str: Report text.
    """
    # Symbolic rule: Base facts.
    base = f"Patient age: {patient_data['age']}, symptom: {patient_data['symptom']}."

    # Neural fluency.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer(base + " Diagnosis:", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=80)
    return tokenizer.decode(outputs[0])


if __name__ == "__main__":
    # Sample case study.
    sample_patient = {"age": 40, "symptom": "fever"}
    print("Medical Report:", medical_nlg(sample_patient))
    # Warning: For research only; add real rules for accuracy in practice.
