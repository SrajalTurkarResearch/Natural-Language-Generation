# Research: Sarcasm Control Scheme for Language Models

## Objective

Design and evaluate a control scheme for a novel attribute—**sarcasm**—in language models, enabling dynamic generation of sarcastic text (e.g., "Oh, great job!") with a continuous control parameter. This scheme, termed the **Contrastive Inversion Module (CIM)**, integrates embedding control, prefix tuning, and LoRA for precise sarcasm modulation.

---

## Control Scheme: Contrastive Inversion Module (CIM)

### Core Concept

Sarcasm inverts literal meaning (e.g., saying "wonderful" to imply failure). CIM captures this by learning paired embeddings: literal ($$\mathbf{z}_l$$) and sarcastic ($$\mathbf{z}_s$$), then interpolating based on a sarcasm scalar ($$s \in [0,1]$$), where $$s=0$$ is sincere and $$s=1$$ is fully sarcastic.

---

### Components

#### 1. **Embedding Inversion**

- Train a lightweight MLP $$g(\mathbf{z}; \phi)$$ to map literal embeddings to sarcastic ones.
- **Contrastive Loss:**
  $$
  \mathcal{L}_{cont} = \max\left(0, d(\mathbf{z}_s, g(\mathbf{z}_l)) - d(\mathbf{z}_s, \mathbf{z}_l) + m\right)
  $$
  where $$d$$ is cosine distance, $$m = 0.1$$.

#### 2. **Prefix Tuning**

- Prepend a sarcasm soft prompt:
  $$
  \mathbf{p}_s = s \cdot \mathbf{v}_s
  $$
  where $$\mathbf{v}_s$$ is a trainable embedding optimized to induce irony.

#### 3. **LoRA Adapter**

- Use a sarcasm-specific LoRA adapter (rank $$r=8$$) to fine-tune attention layers for contextual irony detection.

#### 4. **PPLM-Inspired Steering**

- At inference, use a sarcasm classifier (e.g., fine-tuned BERT) to gradient-boost ironic token probabilities (e.g., favor "oh wonderful" over "wonderful").

#### 5. **Integration**

- For input $$\mathbf{x}$$, generate literal embedding $$\mathbf{z}_l$$, then compute:
  $$
  \mathbf{z}' = (1-s)\mathbf{z}_l + s \cdot g(\mathbf{z}_l)
  $$
- Decode from $$\mathbf{z}'$$ using the LoRA-augmented model with soft prompt $$\mathbf{p}_s$$.

---

## Training Pipeline

- **Dataset:** Sarcasm_V2, Reddit sarcasm threads (e.g., r/sarcasm), paired with non-sarcastic rewrites (generated via GPT-4).

**Steps:**

1. Pre-train MLP ($$g$$) on paired (literal, sarcastic) embeddings using contrastive loss.
2. Fine-tune LoRA and soft prompt on sarcasm dataset with combined loss:
   $$
   \mathcal{L} = \mathcal{L}_{LM} + 0.5\,\mathcal{L}_{cont} + 0.3\,\mathcal{L}_{sarc}
   $$
   where $$\mathcal{L}_{sarc}$$ is from a sarcasm classifier.

- **Compute:** ~1M additional parameters; trainable on a single GPU (e.g., RTX 3090) in 2–4 hours.

---

## Example

- **Input:** "The service was fast." + Sarcasm: 0.8
- **Output:** "Wow, the service was lightning fast, I barely had time to blink!"

---

## Evaluation Design

### Metrics and Methods

| Aspect               | Metrics/Methods                                                                                                                                                                               | Rationale                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Control Accuracy** | - Sarcasm Detection F1: Test with a held-out classifier (e.g., Muennighoff/Sarcasm).<br>- Inversion Fidelity: Cosine similarity between generated and target sarcastic embeddings ($$>0.7$$). | Ensures sarcasm matches control scalar (Pearson $$r > 0.8$$). |
| **Text Quality**     | - Perplexity (PPL): Compare to baseline (target $$<10\%$$ increase).<br>- BLEU/ROUGE vs. gold sarcastic texts.<br>- Diversity: Type-token ratio ($$>0.5$$).                                   | Maintains fluency and variety.                                |
| **Human Evaluation** | - MTurk ratings ($$n=100$$): Sarcasm intensity (1–5), coherence, naturalness.<br>- A/B Testing: CIM vs. PPLM/LoRA baselines.                                                                  | Captures pragmatic nuance; target Cohen's kappa $$>0.6$$.     |
| **Robustness**       | - Adversarial Testing: Vary context length/domains.<br>- Ablation: Remove CIM components (e.g., no MLP).                                                                                      | Validates generalizability; F1 drop $$<5\%$$.                 |
| **Ethics**           | - Bias Audit: Check for stereotypes using fairness toolkits.<br>- Misuse: Evaluate detectability of sarcasm.                                                                                  | Ensures responsible deployment.                               |

---

## Protocol

- **Data Split:** 80% train, 10% validation, 10% test.
- **Statistical Tests:** T-tests for metric significance ($$p<0.05$$).
- **Expected Outcomes:** CIM improves sarcasm F1 by 15–20% over PPLM/LoRA while maintaining PPL within 10% of baseline.

---

## Significance

This scheme advances controlled generation by addressing non-monotonic attributes like sarcasm, enabling applications in creative writing, social media, or conversational agents. Future work could extend CIM to other pragmatic attributes (e.g., politeness, humor).
Significance
This scheme advances controlled generation by addressing non-monotonic attributes like sarcasm, enabling applications in creative writing, social media, or conversational agents. Future work could extend CIM to other pragmatic attributes (e.g., politeness, humor).
