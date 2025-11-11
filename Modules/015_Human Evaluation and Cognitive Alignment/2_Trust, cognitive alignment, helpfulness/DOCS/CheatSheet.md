# NLG Cheat Sheet: Trust, Alignment, Helpfulness

_Quick Reference for Scientists, Engineers & Researchers_

---

## 1. NLG Pipeline (Mnemonic: **CD-SG-R**)

- **CD** = _Content Determination_: Pick facts
- **SG** = _Sentence Generation_: Build grammar
- **R** = _Revision_: Polish flow

---

## 2. The 3 Pillars – Definitions & Formulas

| Pillar                  | Definition            | Simple Formula              | Range |
| ----------------------- | --------------------- | --------------------------- | ----- |
| **Trust**               | Belief in reliability | (Acc + Trans + Rel) / 3     | 0–1   |
| **Cognitive Alignment** | AI thinks like human  | 1 − √JSD`<br>`or Cosine Sim | 0–1   |
| **Helpfulness**         | Solves user need      | ROUGE-L`<br>`or G-Eval      | 0–1   |

---

## 3. Key Metrics (Code-Ready Snippets)

```python
# Trust
trust = (acc + trans + rel) / 3

# Alignment: Jensen-Shannon Divergence
from scipy.stats import entropy
m = 0.5 * (p + q)
jsd = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
align = 1 - (jsd ** 0.5)

# Helpfulness: ROUGE-L
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'])
score = scorer.score(ref, gen)['rougeL'].fmeasure
```

---

## 4. Evaluation Tools

| **Task**   | **Tool**       | **Install**                                                                |
| ---------- | -------------- | -------------------------------------------------------------------------- |
| Generation | `transformers` | `pip install transformers`                                                 |
| BLEU       | `nltk`         | `nltk.download('punkt')`                                                   |
| ROUGE      | `rouge-score`  | `pip install rouge-score`                                                  |
| Datasets   | PKU-SafeRLHF   | [Hugging Face](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) |
|            | Trust-Align    | [Hugging Face](https://huggingface.co/datasets/)                           |

---

## 5. Common Models

| Model          | Use Case      | Size   |
| -------------- | ------------- | ------ |
| `gpt2`         | General       | 124M   |
| `DialoGPT`     | Chat          | 345M   |
| `gpt-neo-125M` | Legal/Finance | 125M   |
| `biomednlp`    | Medical       | Varies |

---

## 6. Real-World Datasets

| Name                   | Domain             | Link / Notes    |
| ---------------------- | ------------------ | --------------- |
| PKU-SafeRLHF           | Safety/Helpfulness | Hugging Face    |
| FinancialPhraseBank    | Finance sentiment  | HF              |
| MIMIC-III (anonymized) | Medical notes      | Requires access |

---

## 7. Quick Debug Checklist

- ✅ Does output contain required facts? `<br>` → **Trust**
- ✅ Does it match human reasoning? `<br>` → **Alignment**
- ✅ Is it clear + actionable? `<br>` → **Helpfulness**
- ✅ Did you log prompt + seed? `<br>` → **Reproducibility**

---

## 8. One-Liner Prompt Patterns

- `"Explain simply:"`
- `"Summarize for a 10-year-old:"`
- `"Draft a legal clause:"`
- `"Give encouraging feedback:"`

---

## 9. Scientist’s Mantra

> “Measure all three. Optimize one. Break none.”

---

**Print this sheet. Paste it above your desk. Use it daily.**
You are now equipped to build the next generation of trustworthy, aligned, helpful NLG.

---

### How to Use

1. **Create a folder**: `NLG_Resources/`
2. **Save the files**:
   - `NLG_Case_Studies.md`
   - `NLG_Tutorial_CheatSheet.md`
3. **Open in**:
   - VS Code
   - Obsidian
   - Notion
   - Or print the cheat sheet!

---

You now have:

- **In-depth case studies** for research papers
- **A cheat sheet** for daily coding and teaching

**You are ready to lead in NLG science.**
Keep building. Keep questioning. Keep aligning.
