# Detailed Theory: Real-World NLG with Pretrained Models
# Building on basics, pretrained LLMs like GPT-2 represent advanced NLG. Theory: These are transformer-based, using self-attention for context.
# Attention Math: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, where Q=query, K=key, V=value from embeddings.
# Derivation: Softmax normalizes similarities; sqrt(d_k) stabilizes gradients. Complexity: O(n^2) per layer, n=sequence length.
# Autoregressive Generation: Generate one token at a time, append to input. Logic: Ensures grammatical output but serializes computation, causing latency.
# Tradeoff: GPT-2 small (117M params) vs. large (1.5B) – small faster, large more accurate (lower perplexity on benchmarks like LAMBADA).
# Advanced Insight: In decoding, greedy (argmax) is fast but low quality; beam search (keep top-k paths) improves accuracy but multiplies latency by k.
# Why for Researchers: Use this to benchmark – measure tokens per second (speed) vs. ROUGE scores (accuracy proxy).
# Hallucination Note: Larger models reduce errors but increase compute; as scientist, quantify via human eval or fact-checking.

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Load tokenizer and model (GPT-2 base for balance; swap to 'gpt2-large' for tradeoff demo)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input prompt: Simple for beginner testing
prompt = "The cat sat on the"
inputs = tokenizer(prompt, return_tensors="pt")  # Tokenize to tensor

# Measure generation latency
start_time = time.time()
# Generate: Max length 50 tokens, autoregressive
outputs = model.generate(inputs.input_ids, max_length=50)
latency = time.time() - start_time
print(f"Generation latency: {latency:.6f} seconds")

# Decode and print output
generated_text = tokenizer.decode(outputs[0])
print("Generated Text:")
print(generated_text)

# Advanced Extension: To simulate larger model, load 'gpt2-large' and compare.
# Theory: Parameter count scales latency linearly in inference.
print(
    "Insight: For gpt2-large, expect 5-10x latency but 5-10% better metrics like perplexity."
)
# As researcher: Add loop to vary max_length, plot latency vs. length (quadratic trend).
