# Advanced Image-to-Story Generator
# Uses BLIP for image-to-text, then GPT-like LLM with CoT for story generation.
# Includes retrieval from a mock knowledge base for plot ideas.
# Author: A hybrid intellect - Turing's logic, Einstein's intuition, Tesla's invention.

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from PIL import Image
import requests
from io import BytesIO

# Load models (assume pre-downloaded; in practice, handle exceptions for loading)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_caption = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_llm = AutoModelForCausalLM.from_pretrained("gpt2")

# Mock retrieval database (advanced: replace with FAISS vector store for semantic search)
knowledge_base = {
    "forest": "A mystical woodland adventure with hidden treasures.",
    "city": "Urban thriller involving espionage and high-stakes chases.",
}


def retrieve_theme(caption):
    """
    Simple keyword-based retrieval; advanced variant: Use cosine similarity on embeddings.
    """
    for key in knowledge_base:
        if key in caption.lower():
            return knowledge_base[key]
    return "Generic story prompt."


def generate_caption(image_url):
    """
    Fetch and caption image using BLIP model.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    inputs = processor(image, return_tensors="pt")
    out = model_caption.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def generate_story(caption, refine=True):
    """
    Generate story via CoT prompting; optional self-refinement loop.
    Mathematical note: Refinement can be seen as iterative optimization, minimizing 'incoherence' (e.g., via entropy H = -sum p log p).
    """
    theme = retrieve_theme(caption)
    prompt = f"Chain-of-Thought: Step 1: Analyze caption: {caption}. Step 2: Retrieve theme: {theme}. Step 3: Outline plot. Step 4: Generate story.\nStory:"
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model_llm.generate(**inputs, max_length=300)
    story = tokenizer.decode(out[0], skip_special_tokens=True)

    if refine:
        refine_prompt = f"Refine this story for coherence: {story}\nRefined:"
        inputs_refine = tokenizer(refine_prompt, return_tensors="pt")
        out_refine = model_llm.generate(**inputs_refine, max_length=300)
        story = tokenizer.decode(out_refine[0], skip_special_tokens=True)

    return story


# Example usage (run as script)
if __name__ == "__main__":
    image_url = "https://example.com/image.jpg"  # Replace with actual URL
    caption = generate_caption(image_url)
    story = generate_story(caption)
    print(f"Caption: {caption}\nStory: {story}")
