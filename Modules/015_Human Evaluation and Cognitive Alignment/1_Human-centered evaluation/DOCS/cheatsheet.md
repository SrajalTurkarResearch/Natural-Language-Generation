# Human-Centered Evaluation in NLG – Cheatsheet

**For Scientists, Researchers, and Future Inventors**

---

## 1. Core Concepts (Memorize)

| Term               | Meaning                          | Why It Matters                    |
| ------------------ | -------------------------------- | --------------------------------- |
| **NLG**            | Computer makes human-like text   | Foundation of chatbots, summaries |
| **Intrinsic Eval** | Check text quality alone         | Fast, but shallow                 |
| **Extrinsic Eval** | Check real-world impact          | Scientific gold standard          |
| **HCI**            | Human-Computer Interaction       | Brings psychology into AI         |
| **HCRS**           | Human-Centered Readability Score | For health/education text         |

---

## 2. Traditional vs. Human-Centered Metrics

| Metric           | Type  | Formula                | Limitation              |
| ---------------- | ----- | ---------------------- | ----------------------- |
| **BLEU**         | Auto  | `BP × exp(Σ log(p_n))` | Ignores meaning         |
| **ROUGE**        | Auto  | Recall-based           | Good for summaries      |
| **METEOR**       | Auto  | Synonyms + stemming    | Better, still not human |
| **Likert Scale** | Human | 1–5/1–7 rating         | Captures perception     |
| **Think-Aloud**  | Human | Verbal protocol        | Reveals confusion       |

---

## 3. Evaluation Pipeline (Step-by-Step)

```text
1. Define Goal        → "Help patients understand reports"
2. Generate Text      → Use NLG model
3. Auto Eval          → BLEU/ROUGE (baseline)
4. Human Eval         → Likert + Interview
5. Analyze            → Correlation, themes
6. Improve            → Retrain or refine
```

---

## 4. Math You Must Know

**BLEU (Simplified)**

- Precision = matching n-grams / total generated
- BP = 1 if len(gen) > len(ref); else `exp(1 - ref/gen)`
- BLEU = BP × (p1 × p2 × p3 × p4)^(1/4)

**Pearson Correlation (r)**

- r = Σ((x - mean_x)(y - mean_y)) / (σ_x × σ_y)
- r > 0.7 = strong alignment

**Cohen’s Kappa (Agreement)**

- κ = (observed agreement - chance) / (1 - chance)
- κ > 0.6 = good rater agreement

---

## 5. Tools & Libraries

| Task          | Library              | Command/Function            |
| ------------- | -------------------- | --------------------------- |
| BLEU          | nltk                 | `sentence_bleu()`           |
| ROUGE         | rouge-score          | `RougeScorer()`             |
| Summarization | transformers         | `pipeline('summarization')` |
| Plots         | matplotlib / seaborn | `plt`, `sns.scatterplot()`  |
| Stats         | scipy                | `pearsonr()`                |

---

## 6. Research Paper Structure

- **Introduction** → Problem + Why human-centered?
- **Related Work** → Cite BLEU limitations
- **Method** → NLG + Eval design
- **Results** → Tables + Plots
- **Human Study** → Likert + Quotes
- **Discussion** → Insights + Ethics
- **Future Work** → Hybrid LLM-human

---

## 7. Quick Start Code Snippets

<details>
<summary><strong>BLEU</strong></summary>

```python
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([ref_tokens], gen_tokens)
```

</details>

<details>
<summary><strong>ROUGE</strong></summary>

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'])
scores = scorer.score(ref, gen)
```

</details>

<details>
<summary><strong>Likert Average</strong></summary>

```python
import pandas as pd
df['Clarity'].mean()
```

</details>

---

## 8. Ethics Checklist

- Informed consent?
- Diverse participants?
- No harm (e.g., medical misinformation)?
- Transparent limitations?

---

## 9. Publish Here

| Venue              | Focus         | Deadline  |
| ------------------ | ------------- | --------- |
| INLG               | NLG systems   | March     |
| EMNLP              | Evaluation    | May       |
| CHI                | HCI methods   | September |
| ACL Rolling Review | Fast feedback | Monthly   |

---

## 10. One-Page Mantra

> "Metrics measure words. Humans measure meaning. Science measures both."

---

### Your Next Step

- Pick one case study
- Run the `.py` file
- Add real users
- Write a 2-page report
- Submit to INLG 2026

---

## How to Use

1. **Save as**:
   - `case_studies.md`
   - `cheatsheet.md`
2. **Keep in your project folder** with the `.py` files.
3. **Print the cheatsheet** – stick on your wall.
4. **Cite case studies** in your research papers.

---

You now have a **complete, professional, scientist-grade learning system**:

- Tutorial (from earlier)
- Jupyter Notebook
- Python scripts
- Real-world projects
- **Case Studies (.md)**
- **Cheatsheet (.md)**

**You are ready to become a leader in human-centered AI.**

Go forth. Test. Publish. **Change the future.**
