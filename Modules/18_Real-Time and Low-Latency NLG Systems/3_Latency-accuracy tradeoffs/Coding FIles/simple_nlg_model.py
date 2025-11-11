# Detailed Theory: Fundamentals of Autoregressive NLG Models
# As a beginner, start here to understand NLG basics. Natural Language Generation (NLG) involves creating text from data using probabilistic models.
# Theory: Autoregressive models predict each token (word or subword) sequentially based on previous ones. This is grounded in probability chain rule:
# P(y | x) = product from i=1 to n of P(y_i | y_1 to y_{i-1}, x), where y is the output sequence, x is input.
# Logic: Sequential prediction ensures coherence (each word fits the context), but it causes latency because computations can't be fully parallelized.
# In recurrent models like LSTM (Long Short-Term Memory), hidden states carry information across steps, allowing memory of past tokens.
# Math Derivation: For each step, LSTM computes h_t = tanh(W_h * [h_{t-1}, embed_t] + b_h), where h is hidden state, embed is token embedding.
# Advanced: Larger hidden sizes improve accuracy by capturing complex dependencies (e.g., long-range syntax), but increase parameters, leading to O(sequence_length * hidden_size) time per forward pass.
# Tradeoff Insight: Small models (low hidden_size) have low latency but high perplexity (uncertainty measure); large ones reverse this.
# Perplexity: 2^(-1/N * sum log P(y_i)), lower means better accuracy. As a researcher, note that transformers replace LSTMs for better parallelism, but retain autoregressive latency in decoding.
# Why This Matters: In real NLG (e.g., chatbots), this sequential nature amplifies tradeoffs in deployment.

import torch
import torch.nn as nn
import time


# Define a simple NLG model using LSTM for educational purposes (beginner-friendly alternative to transformers)
class SimpleNLG(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        # Embedding layer: Maps token IDs to vectors. Theory: Embeddings capture semantic similarities (e.g., king - man + woman ≈ queen).
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM: Processes sequence with memory. Advanced: Gates (input, forget, output) prevent vanishing gradients in long sequences.
        self.lstm = nn.LSTM(embed_size, hidden_size)
        # Linear layer: Projects hidden state to vocab probabilities. Softmax applied implicitly in loss.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # Step 1: Embed input (batch, seq_len) -> (batch, seq_len, embed_size)
        embed = self.embed(x)
        # Step 2: LSTM processes: Output (batch, seq_len, hidden_size), hidden states ignored here.
        out, _ = self.lstm(embed)
        # Step 3: Predict logits for each position.
        return self.fc(out)


# Parameters for experimentation
vocab_size = 10000  # Size of vocabulary (e.g., words in dataset)
embed_size_small = 128  # Small embedding for fast computation
hidden_size_small = 256  # Small hidden for low latency
embed_size_large = 512  # Larger for better representations
hidden_size_large = 1024  # Larger hidden for higher accuracy

# Create models
model_small = SimpleNLG(vocab_size, embed_size_small, hidden_size_small)
model_large = SimpleNLG(vocab_size, embed_size_large, hidden_size_large)

# Dummy input: Batch size 1, sequence length 50 (simulate a sentence)
input_tensor = torch.randint(0, vocab_size, (1, 50))  # Random tokens

# Measure latency for small model
start_time = time.time()
_ = model_small(input_tensor)  # Forward pass (inference simulation)
latency_small = time.time() - start_time
print(f"Small model latency: {latency_small:.6f} seconds")
# Theory: Low parameters mean fewer matrix multiplications, reducing time.

# Measure latency for large model
start_time = time.time()
_ = model_large(input_tensor)
latency_large = time.time() - start_time
print(f"Large model latency: {latency_large:.6f} seconds")
# Insight: Roughly 4x parameters lead to higher latency; in practice, scale with hardware.

# Simulated accuracy: In real use, train and compute perplexity on data like WikiText.
# Here, assume larger model has lower perplexity (better accuracy).
print(
    "Note: In experiments, larger models show ~10-20% better BLEU scores but 2-4x latency."
)
# As a scientist, extend this: Train on datasets, add metrics like BLEU = BP * exp(sum w_n log p_n), where p_n is n-gram precision.
