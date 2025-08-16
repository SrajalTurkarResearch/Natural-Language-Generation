# Major Project: GPT-2 with Top-p Sampling
# Uses Hugging Face Transformers for real NLG.
# Install: pip install transformers torch
# As a scientist, test different prompts and p values.

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def generate_with_top_p(prompt, p=0.9, max_length=50):
    """Generate text using GPT-2 with top-p sampling."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=p)
    return tokenizer.decode(outputs[0])


# Run
if __name__ == "__main__":
    prompt = "In a world where AI rules,"
    print("Generated text:", generate_with_top_p(prompt))
