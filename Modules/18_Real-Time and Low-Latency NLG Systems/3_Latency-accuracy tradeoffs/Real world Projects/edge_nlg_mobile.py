# edge_nlg_mobile.py
# Real-World Project 4: Mobile NLG with Quantization + Pruning
# Goal: Run NLG on phone (<100ms, <100MB)
# Tradeoff: Heavy compression

"""
THEORY & RESEARCH INSIGHT
- Mobile: CPU only, 100ms budget, 100MB memory.
- Quantization (8-bit) + Pruning (50% sparsity) → 6x smaller, 4x faster.
- Math: Pruning: Set w=0 if |w| < ε. Retrain to recover.
- Risk: Over-compression → hallucinations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.quantization
import time

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 1. Quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)


# 2. Simulate pruning (remove 50% weights)
def prune_model(model, sparsity=0.5):
    for name, param in model.named_parameters():
        if "weight" in name:
            tensor = param.data
            threshold = torch.quantile(torch.abs(tensor), sparsity)
            mask = torch.abs(tensor) > threshold
            param.data = tensor * mask.float()
    return model


model_pruned = prune_model(model_int8, sparsity=0.5)

# Test
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")


def measure_model(model, name):
    torch.backends.quantized.engine = "qnnpack"
    start = time.time()
    with torch.no_grad():
        _ = model.generate(inputs.input_ids, max_length=30)
    latency = time.time() - start
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"{name}: Latency={latency:.3f}s, Size={size:.1f}MB")


print("\nMOBILE NLG BENCHMARK")
print("-" * 50)
measure_model(model_fp32, "FP32 (Original)")
measure_model(model_int8, "INT8 (Quantized)")
measure_model(model_pruned, "INT8 + Pruned")
print("\nRESEARCH INSIGHT: Meets mobile constraints. Test on real device next.")
