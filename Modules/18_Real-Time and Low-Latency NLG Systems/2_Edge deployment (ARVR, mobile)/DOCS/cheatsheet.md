## 2. `edge_nlg_cheatsheet.md`

# Edge NLG Cheatsheet – 2025 Edition

> **One-Page Mastery for Scientists & Researchers**  
> _Natural Language Generation on AR/VR/Mobile | November 11, 2025_

---

## 1. Core Theory

| Concept            | Formula                                                    | Meaning                        |
| ------------------ | ---------------------------------------------------------- | ------------------------------ |
| **NLG Generation** | $P(\mathbf{Y} \mid X) = \prod_t P(y_t \mid y_{<t}, X)$     | Autoregressive word prediction |
| **Attention**      | $\mathrm{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$ | Focus on relevant context      |
| **Edge Latency**   | $L_\text{edge} = T_\text{proc}$                            | No network delay               |
| **Cloud Latency**  | $L_\text{cloud} = 2T_\text{send} + T_\text{proc}$          | Round-trip cost                |

---

## 2. Optimization Techniques

| Method           | Math                                                            | Effect                                                    |
| ---------------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| **Pruning**      | $w = 0$ if $\|w\| < \tau$                                       | Sparsity: $s \rightarrow \text{speedup} = \frac{1}{1-s}$  |
| **Quantization** | $q = \mathrm{round}\left(\frac{x-z}{s}\right)$                  | $s = \frac{\mathrm{max}-\mathrm{min}}{2^b-1}$ → 4x memory |
| **Distillation** | $L = \alpha\, \mathrm{CE} + (1-\alpha)\, \mathrm{KL}(T\Vert S)$ | Student learns from teacher                               |
| **SSMs (Mamba)** | $O(L)$ vs $O(L^2)$                                              | Linear scaling for long sequences                         |

---

## 3. Hardware & Tools (2025)

| Platform        | Chip                        | Framework               |
| --------------- | --------------------------- | ----------------------- |
| **Mobile**      | Snapdragon 8 Gen 3, A18 Pro | PyTorch Mobile, ONNX    |
| **AR Glasses**  | Snapdragon AR2              | Qualcomm AI Stack       |
| **VR**          | Quest 3, Jetson Orin        | Unity + TensorFlow Lite |
| **Edge Server** | NVIDIA Orin                 | Triton Inference Server |

---

## 4. Key Models for Edge

| Model            | Params | Quantized Size | Use Case         |
| ---------------- | ------ | -------------- | ---------------- |
| **Llama 3.2 1B** | 1.1B   | 320 MB (4-bit) | Mobile assistant |
| **Gemma 2 2B**   | 2B     | 500 MB         | AR narration     |
| **Mamba-2 1.3B** | 1.3B   | 400 MB         | VR instructions  |
| **DistilGPT2**   | 82M    | 90 MB          | Prototyping      |

---

## 5. Evaluation Metrics

| Metric         | Formula                                                 | Ideal                |
| -------------- | ------------------------------------------------------- | -------------------- |
| **BLEU**       | $\mathrm{BP} \times \exp\left(\sum w_n \log p_n\right)$ | $> 0.6$              |
| **ROUGE-L**    | F1 of longest common subsequence                        | $> 0.5$              |
| **Perplexity** | $\exp(\mathrm{CE})$                                     | $< 20$               |
| **Latency**    | $\mathrm{TTFT} + \mathrm{TPS}$                          | $< 300\,\mathrm{ms}$ |
| **Power**      | mW/token                                                | $< 50\,\mathrm{mW}$  |

---

## 6. Code Snippets

```python
# Quantize Model
model_q = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Generate text
outputs = model_q.generate(
    inputs, max_length=100, temperature=0.7, do_sample=True
)
```

```

```
