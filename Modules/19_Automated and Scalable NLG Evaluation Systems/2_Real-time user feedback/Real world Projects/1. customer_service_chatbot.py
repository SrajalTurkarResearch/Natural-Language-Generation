# customer_service_chatbot.py
# --------------------------------------------------------------
# Real-world: Customer Support – Real-time feedback loop
# --------------------------------------------------------------

import gradio as gr
from transformers import pipeline

# Load a small, fast model (you can replace with any HF model)
generator = pipeline(
    "text-generation", model="distilgpt2", max_length=150, truncation=True
)

# Conversation history (keeps context)
history = ""


def respond(user_msg, feedback=""):
    global history

    # Build prompt: previous context + new user message + optional feedback
    prompt = f"{history}\nCustomer: {user_msg}\n"
    if feedback:
        prompt += f"Feedback: {feedback}\n"
    prompt += "Agent:"

    # Generate
    result = generator(prompt, do_sample=True, top_p=0.9)[0]["generated_text"]
    # Extract only the new agent line
    agent_reply = result.split("Agent:")[-1].strip()

    # Update history
    history = f"{history}\nCustomer: {user_msg}\nAgent: {agent_reply}"
    return agent_reply


# Gradio UI
iface = gr.Interface(
    fn=respond,
    inputs=[
        gr.Textbox(label="Your message", placeholder="How do I reset my password?"),
        gr.Textbox(
            label="Feedback (optional)", placeholder="Make it shorter / Add steps"
        ),
    ],
    outputs=gr.Textbox(label="Agent reply"),
    title="Real-time Customer-Service Chatbot",
    description="Ask a question, then give feedback to improve the answer instantly.",
)

if __name__ == "__main__":
    iface.launch()
