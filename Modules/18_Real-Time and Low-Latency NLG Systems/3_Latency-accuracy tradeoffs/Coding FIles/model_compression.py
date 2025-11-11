# Detailed Theory: Optimization Techniques for Tradeoff Management
# Focus: Model Compression – Reduce size without much accuracy loss.
# Quantization Theory: Convert floats (32-bit) to ints (8-bit), cutting memory/latency.
# Math: w_q = round((w - zero_point) / scale), then dequantize for ops.
# Derivation: Scale = (max_w - min_w) / (2^bits - 1), zero_point shifts range.
# Logic: Reduces precision but preserves distribution; <1% accuracy drop typical.
# Advanced: Dynamic quantization (per-layer) vs. static; in NLG, affects rare tokens more.
# Tradeoff Insight: 4x speed-up possible, but over-quantize causes quantization noise (errors in probabilities).
# Why for Researchers: Use this to deploy – e.g., on edge devices, balance via metrics.
# Pitfall: Retraining (QAT) needed for best results; here, use PyTorch dynamic for simplicity.

from transformers import GPT2LMHeadModel
import torch.quantization
import time

# Load base model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Apply dynamic quantization: Targets linear layers to int8
# Theory: Dynamic means scales computed at runtime, good for variable inputs.
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Test input (from tokenizer, but simplify here)
input_ids = torch.tensor([[50256]])  # Dummy BOS token

# Measure original latency
start_time = time.time()
_ = model.generate(input_ids, max_length=20)
latency_original = time.time() - start_time
print(f"Original model latency: {latency_original:.6f} seconds")

# Measure quantized latency
start_time = time.time()
_ = model_quantized.generate(input_ids, max_length=20)
latency_quantized = time.time() - start_time
print(f"Quantized model latency: {latency_quantized:.6f} seconds")

# Insight: Expect 2-4x speedup; accuracy drop measurable via perplexity on val set.
print(
    "Theory Note: Quantization reduces bit-width, cutting compute by 4x theoretically."
)
# As researcher: Add full eval loop with datasets library, compute BLEU pre/post.
# Advanced: Explore pruning (remove weights < threshold) combined with this.
