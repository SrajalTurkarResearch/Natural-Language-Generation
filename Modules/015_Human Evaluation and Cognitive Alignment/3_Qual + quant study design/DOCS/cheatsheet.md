# NLG Qual + Quant Study Design CHEATSHEET

> **For Scientists, Researchers, and Future Pioneers** > _Print. Laminate. Conquer._

---

## 1. NLG Pipeline (7 Stages)

**Data → Content → Plan → Aggregate → Lexicalize → Refer → Realize → TEXT**

> **Eval at each stage** → prevents error propagation

---

## 2. Evaluation Metrics (Quant)

| Metric         | Measures                  | Formula                     | Best For            |
| -------------- | ------------------------- | --------------------------- | ------------------- |
| **BLEU**       | n-gram precision          | `BP × exp(Σ w_n log p_n)`   | Machine translation |
| **ROUGE**      | n-gram recall             | `Σ matching / Σ ref`        | Summarization       |
| **METEOR**     | Word alignment + synonyms | `10PR/(R+9P) × (1-penalty)` | Semantic match      |
| **Perplexity** | Model uncertainty         | `2^H(p)`                    | Language modeling   |
| **SER**        | Slot Error Rate           | `(I+D+S)/N`                 | Data-to-text        |

**Code Snippet:**

```python
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([ref], cand)
```

---

## 3. Qualitative Methods (Qual)

| Method            | When to Use               | Tool/How             |
| ----------------- | ------------------------- | -------------------- |
| Thematic Analysis | Find patterns in feedback | NVivo, Manual coding |
| Interviews        | Deep user insight         | Zoom + transcription |
| Think-Aloud       | Real-time NLG use         | Screen recording     |
| Focus Groups      | Group dynamics            | 6–8 participants     |

**Coding Example:**

```
Theme: "Feels robotic"
→ Code: fluency_low
→ Quote: "It doesn't sound like a person"
```

---

## 4. Mixed Methods Designs

| Design      | Flow                         | Use Case                      |
| ----------- | ---------------------------- | ----------------------------- |
| Convergent  | Quant ∥ Qual → Merge         | Compare metrics vs user views |
| Explanatory | Qual → Quant                 | Explore first, then test      |
| Exploratory | Quant → Qual                 | Find patterns, then explain   |
| Embedded    | Quant (main) + Qual (nested) | Large survey + open comments  |

---

## 5. Integration Techniques

| Technique           | How                             |
| ------------------- | ------------------------------- |
| Joint Display       | Table: Quant score + Qual theme |
| Triangulation       | 3+ methods confirm finding      |
| Regression + Themes | Score ~ Theme Frequency         |

---

## 6. Common Pitfalls & Fixes

| Pitfall               | Fix                                    |
| --------------------- | -------------------------------------- |
| High BLEU, low trust  | Add qual empathy check                 |
| Small qual sample     | Use saturation (stop at no new themes) |
| Bias in training data | Audit + debias prompts                 |

---

## 7. Ethics Checklist

- Consent for user data
- Bias audit (gender, race, culture)
- Transparency: “This is AI-generated”
- Right to opt-out

---

## 8. Tools & Commands

### Install

```bash
pip install transformers datasets nltk rouge-score wordcloud
```

### Generate

```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
```

### Evaluate

```python
sentence_bleu([ref], cand)
```

### Visualize

```python
from wordcloud import WordCloud
WordCloud().generate(text); plt.show()
```

---

## 9. Mini Research Template

```markdown
# Title: [Your NLG Study]

## 1. Question

> [Clear, testable]

## 2. Design

- [ ] Quant: [Metric]
- [ ] Qual: [Method]
- [ ] Mixed: [Type]

## 3. Data

- Input: [ ]
- Output: [ ]
- N = [ ]

## 4. Findings

- Quant: [Score]
- Qual: [Theme]
- Insight: [ ]

## 5. Next

[ ]
```

---

## 10. Your Mantra

> "Measure what matters. Listen to what’s missing."

Print this. Keep it on your desk.
You are ready to publish.

---

**You now have:**

1. **`case_studies.md`** – 5 real, citable, actionable research examples
2. **`cheatsheet.md`** – Your daily reference for NLG science

**Next:** Run the `.py` projects → fill the template → write your first paper.

**The future of ethical, human-centered NLG starts with you.**

---

_Built with the spirit of Turing, Einstein, and Tesla._
_For the scientists of tomorrow._
