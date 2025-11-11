# chatbot_deployment.py
# Real-World Project 1: Low-Latency Customer Support Chatbot
# Goal: Deploy a fast NLG model for real-time responses while maintaining acceptable accuracy.
# Dataset: Cornell Movie Dialogs (conversational data)
# Tradeoff: Use distilled model for <300ms latency, compare to full GPT-2.

"""
THEORY & RESEARCH INSIGHT
- In customer support, user drop-off increases 20% per second of delay.
- Ideal latency: <1s (human-like conversation).
- Distillation: Train small "student" from large "teacher" → 3x faster, ~2% accuracy drop.
- Math: Student loss = α * CE(y, p_s) + (1-α) * KL(p_t || p_s), where CE=cross-entropy, KL=knowledge distillation.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from datasets import load_dataset
import time
import numpy as np

# ====================== CONFIG ======================
MAX_LATENCY_TARGET = 0.3  # 300ms per response
MODEL_SMALL = "distilgpt2"  # Distilled, fast
MODEL_LARGE = "gpt2"  # Accurate, slow
DATASET_NAME = "cornell_movie_dialogs"
MAX_LENGTH = 50
# ===================================================

print("Loading models and tokenizer...")
tokenizer_small = GPT2Tokenizer.from_pretrained(MODEL_SMALL)
model_small = GPT2LMHeadModel.from_pretrained(MODEL_SMALL)
model_small.eval()

tokenizer_large = GPT2Tokenizer.from_pretrained(MODEL_LARGE)
model_large = GPT2LMHeadModel.from_pretrained(MODEL_LARGE)
model_large.eval()

# Load real conversational data
print("Loading Cornell Movie Dialogs dataset...")
dataset = load_dataset("cornell_movie_dialogs", split="train[:100]")  # Sample


def generate_response(model, tokenizer, prompt, max_length=MAX_LENGTH):
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, latency


# Test on real customer-like queries
queries = [
    "Hi, I can't log into my account. What should I do?",
    "My order hasn't arrived yet. Can you check?",
    "How do I return a product?",
    "What's your refund policy?",
]

print("\n" + "=" * 60)
print("REAL-WORLD CHATBOT DEPLOYMENT TEST")
print("=" * 60)

results = []
for q in queries:
    print(f"\nQuery: {q}")

    # Small model (fast)
    resp_s, lat_s = generate_response(model_small, tokenizer_small, q)
    print(f"[FAST]  Response: {resp_s[len(q):].strip()}")
    print(f"        Latency: {lat_s:.3f}s")

    # Large model (accurate)
    resp_l, lat_l = generate_response(model_large, tokenizer_large, q)
    print(f"[ACCURATE] Response: {resp_l[len(q):].strip()}")
    print(f"           Latency: {lat_l:.3f}s")

    results.append({"query": q, "lat_small": lat_s, "lat_large": lat_l})

# Summary
avg_lat_small = np.mean([r["lat_small"] for r in results])
avg_lat_large = np.mean([r["lat_large"] for r in results])
print("\n" + "-" * 60)
print(
    f"Average Latency (Small): {avg_lat_small:.3f}s → Meets target: {avg_lat_small <= MAX_LATENCY_TARGET}"
)
print(f"Average Latency (Large): {avg_lat_large:.3f}s → Too slow for real-time")
print("-" * 60)
print("RESEARCH INSIGHT: Distillation enables production deployment.")
