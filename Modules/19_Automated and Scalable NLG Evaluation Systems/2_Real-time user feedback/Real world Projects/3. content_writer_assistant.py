# content_writer_assistant.py
# --------------------------------------------------------------
# Real-world: Real-time Content Creation Assistant
# --------------------------------------------------------------

import gradio as gr
from transformers import pipeline

generator = pipeline(
    "text-generation", model="EleutherAI/gpt-neo-125M", max_length=120, truncation=True
)


def rewrite(text, style_feedback=""):
    prompt = f"Original: {text}\n"
    if style_feedback:
        prompt += f"Style instruction: {style_feedback}\n"
    prompt += "Rewritten version:"

    out = generator(prompt, do_sample=True, top_p=0.9)[0]["generated_text"]
    rewritten = out.split("Rewritten version:")[-1].strip()
    return rewritten


iface = gr.Interface(
    fn=rewrite,
    inputs=[
        gr.Textbox(label="Your draft", lines=4, placeholder="The product is good."),
        gr.Textbox(
            label="Style feedback", placeholder="Make it formal / Add enthusiasm"
        ),
    ],
    outputs=gr.Textbox(label="Improved version"),
    title="AI Writing Assistant",
    description="Paste a draft, give style feedback, get an instant rewrite.",
)

if __name__ == "__main__":
    iface.launch()
