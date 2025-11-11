# model_quantization.py
# Purpose: Quantize loaded model for edge efficiency (AR/VR/mobile).
# Theory: Quantization maps floats to ints, reducing model size and speeding inference on low-precision hardware (e.g., mobile NPUs).
#         Error minimized via calibration: MSE(original, dequantized) < threshold.
# Logic: Use torch.quantization.dynamic for runtime conversion; test generation.
# As engineer: Deploy on Android via PyTorch Mobile; measure size reduction.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model (from previous file logic)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Quantize: Dynamic for linear layers (common in transformers), to int8
model_quantized = torch.quantization.quantize_dynamic(
    model,  # Input model
    {torch.nn.Linear},  # Target modules (attention/feedforward layers)
    dtype=torch.qint8,  # 8-bit signed int
)

# Test generation
input_text = "Edge deployment improves"  # Prompt
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model_quantized.generate(inputs["input_ids"], max_length=50)
generated_text = tokenizer.decode(outputs[0])

print("Quantized Generated Text:", generated_text)

# Measure size (approximate): Serialize and check length
import io

buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
original_size = buffer.tell() / (1024 * 1024)  # MB

buffer = io.BytesIO()
torch.save(model_quantized.state_dict(), buffer)
quantized_size = buffer.tell() / (1024 * 1024)

print(
    f"Original Size: {original_size:.2f} MB, Quantized: {quantized_size:.2f} MB"
)  # Expect ~4x reduction
# Research: Fine-tune quantization-aware training (QAT) for better accuracy
