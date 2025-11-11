# mini_project.py
# Purpose: Build a basic edge-ready NLG chatbot.
# Theory: Combines distillation (small model) with quantization for low-resource devices.
# Logic: Interactive loop; generate responses based on prompts.
# As engineer: Deploy via Flask for mobile app backend; measure power with adb.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load and quantize (from previous files)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print("Edge NLG Chatbot: Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    prompt = f"Response to: {user_input}"  # Frame as NLG task
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model_quantized.generate(
        inputs["input_ids"], max_length=100, num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Bot:", response)

# Research: Add metrics like perplexity = exp(cross-entropy) for quality evaluation
