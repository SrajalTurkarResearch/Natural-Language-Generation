import torch
import heapq
from toy_lm import ToyLM, vocab, vocab_size

# Beam Search: Tracks top-K sequences forВП

System: <xaiArtifact artifact_id="e8d3451f-6a40-4f50-a868-fe4da6ff62f3" artifact_version_id="470d1108-2b04-4dc7-a5d3-d68abecc0c8f" title="beam_search.py" contentType="text/python">
import torch
import heapq
from toy_lm import ToyLM, vocab, vocab_size

# Beam Search: Maintains K hypotheses to explore multiple sequence paths
# Essential for high-quality NLG in research, balances speed and accuracy
def beam_search(model, start_token, beam_width=3, max_len=10):
    beams = [([start_token], 0.0)]  # List of (sequence, log_prob)
    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == 7:  # <eos>
                    new_beams.append((seq, score))  # Keep complete sequences
                    continue
                input = torch.tensor([seq])
                probs = model(input).squeeze(0)
                top_k = torch.topk(probs, beam_width)  # Get top K tokens
                for val, idx in zip(top_k.values, top_k.indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + torch.log(val).item()  # Log prob for stability
                    new_beams.append((new_seq, new_score))
            beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[1])  # Keep top K
    best_seq, _ = max(beams, key=lambda x: x[1])
    return best_seq

# Example usage
if __name__ == "__main__":
    model = ToyLM(vocab_size)
    start_token = 0  # <start>
    seq = beam_search(model, start_token)
    print("Beam Search Output:", [vocab[i] for i in seq])