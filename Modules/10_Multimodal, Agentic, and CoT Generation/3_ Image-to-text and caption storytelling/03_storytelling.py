# Caption Storytelling: Generating Narrative Captions
# This file extends captions into stories using GPT-2.

# Install: pip install transformers torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import matplotlib.pyplot as plt

# --- Theory ---
# Storytelling builds on captions by adding narrative elements (start, middle, end) and emotional context.
# GPT-2 is a language model that generates text by predicting the next word based on previous ones.
# We use a caption (from BLIP or manual) as a prompt to create a story.


# --- Code Guide ---
def generate_story(caption="A cat sleeps on a couch"):
    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Create prompt
    prompt = f"Create a short story based on this image description: {caption}. Story:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Story:", story)

    # Visualize (simulated image for context)
    plt.figure(figsize=(5, 5))
    plt.text(0.5, 0.5, story, wrap=True, ha="center", va="center")
    plt.axis("off")
    plt.title("Story Visualization")
    plt.show()


# --- Exercise ---
# 1. Run with a different caption (e.g., "A dog runs in a park").
# 2. Change max_length to 150. Does the story get better or worse? Why?

# --- Research Insight ---
# Storytelling models struggle with coherence over long texts. Recent work (e.g., 2025 papers) uses reinforcement learning to reward coherent narratives. Explore how to measure "coherence" quantitatively.

if __name__ == "__main__":
    print("Running Storytelling")
    generate_story()
