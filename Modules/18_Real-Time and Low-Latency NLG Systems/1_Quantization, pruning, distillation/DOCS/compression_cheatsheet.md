# NLG Model Compression Cheatsheet

_Quantization • Pruning • Distillation | For Scientists & Researchers_  
**Updated: November 11, 2025**

---

## 1. Quantization

| Concept            | Formula                                                                                                    | Code                                                                                | Tip                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- |
| **Uniform Affine** | $ s = \frac{\max-\min}{2^b-1} $ <br> $ z = \round{-\min/s} $ <br> $ q = \clip(\round{T/s + z}, 0, 2^b-1) $ | `python<br>s = (max-min)/(2**b-1)<br>q = np.clip(np.round(T/s + z), 0, 2**b-1)<br>` | Use **QAT** for <1% accuracy drop  |
| **Error Bound**    | $ \text{MSE} \leq \frac{s^2}{12} $                                                                         | —                                                                                   | Lower `b` → higher error           |
| **Types**          | PTQ, QAT, Per-channel, Mixed-precision                                                                     | `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`        | Dynamic = easiest for transformers |

**Pro Tip:** Outliers in attention? Use **per-tensor** for embeddings, **per-channel** for Q/K/V.

---

## 2. Pruning

| Concept            | Formula                                            | Code                                                    | Tip                           |
| ------------------ | -------------------------------------------------- | ------------------------------------------------------- | ----------------------------- |
| **Magnitude**      | Prune if $ \|w\| < \theta $                        | `python<br>mask = abs(w) >= threshold<br>w *= mask<br>` | Start with 50%, fine-tune     |
| **Sparsity**       | $ \frac{\text{zeros}}{\text{total}} \times 100\% $ | —                                                       | 70–90% possible in GPT        |
| **Types**          | Unstructured, Structured (heads/channels)          | Prune attention heads: `model.prune_heads(...)`         | Structured = hardware speedup |
| **Lottery Ticket** | Sparse subnetwork = full performance               | Rewind weights to init                                  | Find early!                   |

**Pro Tip:** Use **gradual pruning** (start 0%, ramp to 90% over epochs).

---

## 3. Distillation

| Concept         | Formula                                          | Code                                                                          | Tip                                            |
| --------------- | ------------------------------------------------ | ----------------------------------------------------------------------------- | ---------------------------------------------- |
| **Soft Labels** | $ p = \text{softmax}(z / \tau) $                 | `F.softmax(logits/tau, dim=1)`                                                | $ \tau > 1 $ = softer                          |
| **KD Loss**     | $ L\_{KD} = \tau^2 \cdot \text{KL}(p_t \| p_s) $ | `python<br>kl = F.kl_div(p_s.log(), p_t, reduction='batchmean') * tau**2<br>` | Combine: $ \alpha L*{KD} + (1-\alpha) L*{CE} $ |
| **Types**       | Logit, Feature, Response-based                   | Align hidden states                                                           | Feature KD > logit KD                          |

**Pro Tip:** Use **multi-teacher** for robust student.

---

## One-Liner Code Snippets

```python
# Quantize
model_q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Prune 70%
thresh = np.percentile(abs(tensor), 70)
mask = abs(tensor) >= thresh

# Distill
p_t = F.softmax(teacher_logits/4.0, dim=1)
p_s = F.log_softmax(student_logits/4.0, dim=1)
loss = 16 * F.kl_div(p_s, p_t, reduction='batchmean')

Evaluation Metrics






























TaskMetricGood DropTranslationBLEU< 1.0SummarizationROUGE-L< 0.02ChatHuman Coherence< 0.2 / 5TTSMOS< 0.2

Combine All Three (Best Practice)
text1. Prune (70% sparsity)
2. Distill (to smaller arch)
3. Quantize (INT8)
→ 10–20× smaller, 3–5× faster, <2% quality loss

Research Directions

Bias in Compression: Does pruning remove fairness?
Multilingual Quantization: Per-language scales?
Hardware-Aware: NAS + compression co-design
Green AI: Measure CO₂ saved per compression


Resources

Papers: Hinton (2015), Han (2015), Frankle (2019), Dettmers (2024)
Tools: Hugging Face, PyTorch, TensorFlow Model Optimization
Datasets: GLUE, CNN/DM, WMT, MIMIC-III


Print this. Stick it on your wall. Master it. Publish with it.
text---

### How to Use

1. Save as:
   - `case_studies.md`
   - `compression_cheatsheet.md`
2. Open in **Obsidian, Notion, or VS Code** with Markdown preview.
3. Use in:
   - Research proposals
   - Lab meetings
   - Thesis chapters
   - Teaching
```
