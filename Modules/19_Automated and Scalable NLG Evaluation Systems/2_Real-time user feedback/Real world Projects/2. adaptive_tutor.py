# adaptive_tutor.py
# --------------------------------------------------------------
# Real-world: Adaptive Tutoring System
# --------------------------------------------------------------

import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", max_length=200, truncation=True)

# Keep track of topic & previous explanation
topic = ""
explanation = ""


def teach(new_topic, feedback=""):
    global topic, explanation

    if new_topic and new_topic != topic:
        # New topic → fresh explanation
        prompt = f"Explain the concept of {new_topic} in simple terms."
        topic = new_topic
    else:
        # Same topic → refine with feedback
        prompt = f"Previous explanation: {explanation}\nFeedback: {feedback}\nImproved explanation of {topic}:"

    out = generator(prompt, do_sample=True, top_p=0.92)[0]["generated_text"]
    # Extract only the new part
    new_exp = (
        out.split("Improved explanation")[-1].strip()
        if "Improved" in out
        else out.strip()
    )
    explanation = new_exp
    return new_exp


iface = gr.Interface(
    fn=teach,
    inputs=[
        gr.Textbox(
            label="Topic (or leave blank to continue)", placeholder="Photosynthesis"
        ),
        gr.Textbox(label="Feedback", placeholder="I don't get it / Give an example"),
    ],
    outputs=gr.Textbox(label="Tutor explanation"),
    title="Adaptive AI Tutor",
    description="Start a topic, then give feedback to make the explanation clearer.",
)

if __name__ == "__main__":
    iface.launch()
