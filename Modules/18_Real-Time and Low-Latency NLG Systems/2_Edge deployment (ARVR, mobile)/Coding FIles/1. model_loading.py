# model_loading.py
# Purpose: Load a small NLG model for edge deployment basics.
# Theory: DistilGPT2 is compressed via knowledge distillation, where a small student model mimics a large teacher's soft probabilities.
#         Loss = α * CE(student, labels) + (1-α) * KL(teacher_probs || student_probs), with KL = ∑ p log(p/q).
# Logic: Load pretrained model and tokenizer; generate sample text to verify.
# As researcher: Measure model size with sys.getsizeof or torchsummary for edge constraints.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model name (DistilGPT2: Efficient for mobile/AR due to fewer layers/heads)
model_name = "distilgpt2"

# Load tokenizer: Converts text to token IDs (vocabulary size ~50k)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model: Causal LM for autoregressive generation
model = AutoModelForCausalLM.from_pretrained(model_name)

# Sample generation to test
input_text = "The basics of NLG are"  # Prompt for generation
inputs = tokenizer(
    input_text, return_tensors="pt"
)  # return_tensors="pt" for PyTorch format
outputs = model.generate(
    inputs["input_ids"], max_length=50
)  # Autoregressive: Predict next token iteratively
generated_text = tokenizer.decode(outputs[0])  # Decode back to text

print("Generated Text:", generated_text)

# Advanced: Add timing for latency measurement (key for edge)
import time

start = time.perf_counter()
outputs = model.generate(inputs["input_ids"], max_length=50)
end = time.perf_counter()
print(
    f"Inference Time: {end - start:.4f} seconds"
)  # Research: Compare on CPU vs. GPU for mobile simulation
