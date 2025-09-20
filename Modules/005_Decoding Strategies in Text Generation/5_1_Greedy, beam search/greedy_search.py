import torch
from toy_lm import ToyLM, vocab, vocab_size


# Greedy Search: Selects the highest-probability token at each step
# Ideal for quick prototyping in research, but may miss optimal sequences
def greedy_search(model, start_token, max_len=10):
    sequence = [start_token]  # Initialize with start token
    with torch.no_grad():  # Disable gradient computation for inference
        for _ in range(max_len):
            input = torch.tensor([sequence])  # Current sequence as input
            probs = model(input)  # Get probabilities [1, vocab_size]
            next_token = torch.argmax(probs, dim=-1).item()  # Pick highest prob
            sequence.append(next_token)
            if next_token == 7:  # Stop at <eos>
                break
    return sequence


# Example usage
if __name__ == "__main__":
    model = ToyLM(vocab_size)
    start_token = 0  # <start>
    seq = greedy_search(model, start_token)
    print("Greedy Search Output:", [vocab[i] for i in seq])
