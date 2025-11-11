# medical_report_generator.py
# --------------------------------------------------------------
# Real-world: Clinical Report Generation with Doctor Feedback
# --------------------------------------------------------------

import gradio as gr
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    max_length=180,
    truncation=True,
)

patient_data = "Age: 55, Symptoms: chest pain, BP: 150/90, ECG: ST elevation"


def generate_report(feedback=""):
    base_prompt = f"Patient data: {patient_data}\n"
    if feedback:
        base_prompt += f"Doctor feedback: {feedback}\n"
    base_prompt += "Clinical report:"
    out = generator(base_prompt, do_sample=True, top_p=0.88)[0]["generated_text"]
    report = out.split("Clinical report:")[-1].strip()
    return report


iface = gr.Interface(
    fn=generate_report,
    inputs=gr.Textbox(
        label="Doctor feedback (optional)",
        placeholder="Add risk factors / Use simpler terms",
    ),
    outputs=gr.Textbox(label="Generated clinical report", lines=4),
    title="AI Medical Report Writer",
    description="Doctor reviews auto-generated report and gives instant feedback.",
)

if __name__ == "__main__":
    iface.launch()
