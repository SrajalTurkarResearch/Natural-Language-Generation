# Latency-Accuracy Tradeoffs in NLG – Quick Reference Cheatsheet

> **For Aspiring Scientists & Researchers** > _Print, pin, memorize_ — your one-page NLG survival guide.

---

## 1. Core Metrics

| Metric         | Formula                   | Best For          | Range    |
| -------------- | ------------------------- | ----------------- | -------- |
| **BLEU**       | `BP × exp(Σ w_n log p_n)` | N-gram overlap    | 0–1      |
| **ROUGE-L**    | `LCS(X,Y) / len(Y)`       | Long sequences    | 0–1      |
| **Perplexity** | `2^(-1/N Σ log P(y_i))`   | Model uncertainty | ↓ better |
| **Latency**    | `time(end) - time(start)` | Speed             | ms/s     |

---

## 2. Tradeoff Laws

```text
Accuracy ∝ log(Parameters)
Latency  ∝ Parameters
→ A = c × log(L/k)  → Diminishing returns
```

---

## 3. Optimization Techniques

| Technique                | Speedup | Accuracy Drop | Use When            |
| ------------------------ | ------- | ------------- | ------------------- |
| **Distillation**         | 3–5x    | 1–3%          | Production chatbots |
| **Quantization (INT8)**  | 2–4x    | <1%           | Mobile/Edge         |
| **Pruning**              | 2–3x    | 1–5%          | Memory-constrained  |
| **Speculative Decoding** | 2–3x    | 0%            | High-throughput     |
| **Beam Search (k=4)**    | —       | +5–10%        | Medical/Legal       |

---

## 4. Decoding Strategies

| Strategy        | Latency | Accuracy | Command         |
| --------------- | ------- | -------- | --------------- |
| Greedy          | Fast    | Low      | do_sample=False |
| Top-p (Nucleus) | Medium  | Good     | top_p=0.9       |
| Beam Search     | Slow    | High     | num_beams=4     |

---

## 5. Hardware Impact

| Device     | FLOPs/s    | Ideal Model   |
| ---------- | ---------- | ------------- |
| GPU (A100) | 312 TFLOPs | GPT-2 Large   |
| CPU (Xeon) | 1 TFLOPs   | DistilGPT2    |
| Mobile CPU | 10 GFLOPs  | INT8 + Pruned |

---

## 6. Quick Commands (Hugging Face)

bash

```
# Generate
model.generate(input_ids, max_length=50, do_sample=True, top_p=0.9)

# Quantize
torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Evaluate BLEU
from sacrebleu import sentence_bleu
bleu = sentence_bleu(hypothesis, [reference]).score
```

---

## 7. Pareto Frontier Decision Rule

python

```
if latency_budget < 300ms:
    use_distilled_model()
elif accuracy_needed > 95%:
    use_beam_search()
else:
    use_adaptive_rag()
```

---

## 8. Research Checklist

- Measure **both** latency and accuracy
- Plot **Pareto curve**
- Use **real dataset** (NQ, MIMIC, etc.)
- Report **hardware specs**
- Include **human evaluation**

---

> **Pro Tip** : In interviews, say:
> _"I optimize NLG via distillation and adaptive retrieval, achieving 45% latency reduction with <2% accuracy loss on Natural Questions."_

---

text

```
---

## How to Use These Files

1. **Save as**:
   - `case_studies.md`
   - `cheatsheet.md`

2. **In Your Workflow**:
   - Cite `case_studies.md` in **papers, reports, presentations**
   - Print `cheatsheet.md` and **keep on your desk**

3. **Next Step**:
   > Ask me:
   > **"Help me write a 2-page research poster using these case studies."**

```
