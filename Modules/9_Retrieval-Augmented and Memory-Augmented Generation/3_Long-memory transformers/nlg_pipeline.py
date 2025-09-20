# nlg_pipeline.py: Long-Memory NLG Pipeline for Text Generation
# Author: Grok, inspired by Turing, Einstein, Tesla
# Purpose: Build end-to-end NLG with HMT, generate coherent long text
# Theory: Memory as a library—HMT indexes past for future generation
# Application: Story continuation remembering early plot points

import torch
import torch.nn as nn
import torch.nn.functional as F


# Reuse SimpleAttention and SimpleHMT (imagine importing from hmt_model.py)
class SimpleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = X.shape
        Q = (
            self.W_q(X)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(X)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(X)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.W_o(attn_out)


class SimpleHMT(nn.Module):
    def __init__(
        self, d_model: int, L: int = 1024, k: int = 32, N: int = 300, j: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.L = L
        self.k = k
        self.N = N
        self.j = j
        self.prompt_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.backbone = SimpleAttention(d_model)
        self.memory_bank = []

    def summarize(self, H_n: torch.Tensor) -> torch.Tensor:
        T = self.prompt_embed.expand(H_n.size(0), -1, -1)
        input_sum = torch.cat([T, H_n[:, : self.j, :], T], dim=1)
        S_n = self.backbone(input_sum)[:, self.j, :]
        return S_n.unsqueeze(1)

    def retrieve(self, S_n: torch.Tensor) -> torch.Tensor:
        if not self.memory_bank:
            return torch.zeros_like(S_n)
        M_past = torch.stack(self.memory_bank[-self.N :])
        Q_n = self.W_q(S_n)
        K_past = self.W_k(M_past)
        scores = torch.matmul(Q_n, K_past.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn = F.softmax(scores, dim=-1)
        P_n = torch.matmul(attn, M_past)
        return P_n.squeeze(1)

    def forward(self, segments: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        prev_sensory = None
        for H_n in segments:
            S_n = self.summarize(H_n)
            P_n = self.retrieve(S_n)
            if prev_sensory is not None:
                input_h = torch.cat(
                    [prev_sensory, P_n.unsqueeze(1), H_n, P_n.unsqueeze(1)], dim=1
                )
            else:
                input_h = torch.cat([P_n.unsqueeze(1), H_n, P_n.unsqueeze(1)], dim=1)
            H_out = self.backbone(input_h)[:, self.k + 1 : self.L + self.k + 2, :]
            H_out_n, M_n = H_out[:, :-1, :], H_out[:, -1:, :]
            outputs.append(H_out_n)
            self.memory_bank.append(M_n.squeeze(1))
            prev_sensory = H_n[:, -self.k :, :]
        return outputs


# NLG Pipeline with Long-Memory
class LongMemoryNLG(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1000, d_model))  # Simplified
        self.layers = nn.ModuleList([SimpleHMT(d_model) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(
        self, input_ids: torch.Tensor, generate: bool = False, max_len: int = 50
    ):
        # Embed input with positional encoding
        embedded = self.embed(input_ids) + self.pos_enc[:, : input_ids.size(1), :]
        # Split into segments
        segments = [
            embedded[:, i * 1024 : (i + 1) * 1024, :]
            for i in range((input_ids.size(1) // 1024) + 1)
        ]
        memories = []
        for layer in self.layers:
            layer_outs = layer(segments)
            memories.append([o.mean(dim=1) for o in layer_outs])
        last_mem = memories[-1][-1]
        if generate:
            generated = []
            current = last_mem.unsqueeze(1)
            for _ in range(max_len):
                logits = self.fc_out(current)
                next_id = torch.argmax(logits[:, -1:, :], dim=-1)
                generated.append(next_id)
                current = torch.cat([current, self.embed(next_id)], dim=1)
            return torch.cat(generated, dim=1)
        return self.fc_out(embedded.mean(dim=1))


# Test NLG
def test_nlg():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 100
    model = LongMemoryNLG(vocab_size).to(device)
    input_ids = torch.randint(0, vocab_size, (1, 2000)).to(device)  # Long input
    logits = model(input_ids)
    print("Logits shape:", logits.shape)
    gen_ids = model(input_ids, generate=True)
    print("Generated sequence sample:", gen_ids[0][:10].cpu().numpy())


# Mini Project: Summarize 5k-token text
# Steps: 1. Load text (e.g., NLTK). 2. Chunk, process with HMT. 3. Generate summary
# Dataset: CNN/DailyMail (concat for long text)
def simple_lrmt_summary(text_chunks: List[str], vocab_size=100):
    # Placeholder: Real-world use tokenizer (e.g., Hugging Face)
    return "Summarized text..."


# Major Project: Fine-tune on PG-19 for story generation [Citation ID: 27]
# Steps: 1. Download PG-19 (torchtext). 2. Tokenize segments. 3. Train HMT. 4. Generate
# Metric: Perplexity = exp(-log P). Goal: Drop 15% vs standard
# Research: Ablate hierarchy—does removing long-term hurt by 20%?
if __name__ == "__main__":
    test_nlg()

# Exercise 4: Build mini NLG on NarrativeXL subset; measure forgetting
# Solution: Compare accuracy on early vs late queries; expect 10% drop without HMT
# Research Insight: Like Tesla's coils, HMT channels distant info—test multimodal next
# For notes: Sketch pipeline: Input -> HMT Layers -> Decoder -> Story
