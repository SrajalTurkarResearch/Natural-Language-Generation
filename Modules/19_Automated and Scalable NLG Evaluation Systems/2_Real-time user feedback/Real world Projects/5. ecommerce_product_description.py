# ecommerce_product_description.py
# --------------------------------------------------------------
# Real-world: E-commerce – Update product description from reviews
# --------------------------------------------------------------

import gradio as gr
from transformers import pipeline

generator = pipeline(
    "text-generation", model="distilgpt2", max_length=130, truncation=True
)
sentiment = pipeline("sentiment-analysis")

# Simulated past reviews
reviews = [
    "Super lightweight and comfortable!",
    "Battery lasts only 2 hours.",
    "Great value for money.",
]


def analyze_and_update(current_desc, new_review=""):
    # Add new review
    if new_review:
        reviews.append(new_review)

    # Simple sentiment aggregation
    pos = sum(1 for r in reviews if sentiment(r)[0]["label"] == "POSITIVE")
    neg = len(reviews) - pos
    sentiment_summary = f"{pos} positive, {neg} negative mentions."

    prompt = f"Current description: {current_desc}\nCustomer sentiment: {sentiment_summary}\nImproved description:"
    out = generator(prompt, do_sample=True, top_p=0.9)[0]["generated_text"]
    new_desc = out.split("Improved description:")[-1].strip()
    return new_desc, sentiment_summary


iface = gr.Interface(
    fn=analyze_and_update,
    inputs=[
        gr.Textbox(
            label="Current product description",
            value="Wireless earbuds with noise cancellation.",
            lines=2,
        ),
        gr.Textbox(label="New customer review", placeholder="Too bulky for my ears."),
    ],
    outputs=[
        gr.Textbox(label="Updated description", lines=3),
        gr.Textbox(label="Sentiment summary"),
    ],
    title="Dynamic E-commerce Product Description",
    description="Enter a new review → AI instantly updates the product copy.",
)

if __name__ == "__main__":
    iface.launch()
