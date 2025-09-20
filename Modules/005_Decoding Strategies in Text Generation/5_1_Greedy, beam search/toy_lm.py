import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple toy language model for NLG demonstrations
# As a scientist, you'll use this to understand how models predict token probabilities
class ToyLM(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=5):
        super().__init__()
        # Embedding layer maps tokens to vectors
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Linear layer predicts probabilities over vocabulary
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # Input x: tensor of token indices [batch, sequence_length]
        # Output: probabilities over next token [batch, vocab_size]
        embed = self.embed(x)  # Convert tokens to embeddings
        logits = self.fc(embed.mean(dim=1))  # Aggregate embeddings (simplified)
        return F.softmax(logits, dim=-1)  # Softmax for probabilities


# Vocabulary for examples (maps indices to words)
# Use this to interpret model outputs in your experiments
vocab = ["<start>", "the", "cat", "sat", "on", "mat", "hat", "<eos>", "dog", "runs"]
vocab_size = len(vocab)

# Example usage
if __name__ == "__main__":
    model = ToyLM(vocab_size)
    # Test with a single token input (e.g., <start>)
    input_seq = torch.tensor([[0]])  # <start>
    probs = model(input_seq)
    print("Sample probabilities:", probs)
    print("Predicted next token:", vocab[torch.argmax(probs, dim=-1).item()])
