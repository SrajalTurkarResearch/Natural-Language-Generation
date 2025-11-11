### 2. `cheatsheet.md`

**Purpose**: **One-page reference** for all concepts, formulas, code, and commands — **print it, stick it on your wall**.

```markdown
# EaaS in NLG – Ultimate Cheatsheet

_For Scientists & Researchers_
**Print A4 | Stick on Wall | Master in 5 Minutes**

---

## 1. NLG Pipeline (3 Steps)
```

Data → [Content Planning] → [Sentence Planning] → [Surface Realization] → Text

text

```
- **Plan**: What to say?
- **Structure**: How to say?
- **Realize**: Grammar + style

---

## 2. Evaluation Types
| Type | Measures | Example |
|------|---------|--------|
| **Intrinsic** | Text quality | BLEU, Grammar |
| **Extrinsic** | Real impact | Click rate, Diagnosis speed |

---

## 3. Top 5 Metrics (Formulas)

| Metric | Formula | Best For |
|--------|-------|---------|
| **BLEU** | `BP × exp(Σ log pₙ / 4)` | Translation |
| **ROUGE-L** | `2 × LCS / (len(G) + len(R))` | Summarization |
| **BERTScore** | `Σ max cos(emb_g, emb_r)` | Meaning |
| **METEOR** | Harmony of P+R + synonyms | General |
| **Fairness Gap** | `|E[score|male] - E[score|female]|` | Bias |

---

## 4. EaaS Architecture
```

You → POST /evaluate → Cloud → BLEU, ROUGE, Fairness → JSON + Dashboard

text

````
---

## 5. Python Code Snippets

```python
# BLEU (1 line)
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([ref.split()], gen.split())

# ROUGE
from rouge import Rouge
r = Rouge(); r.get_scores(gen, ref)[0]['rouge-l']['f']

# Mini EaaS API
from flask import Flask, request
app = Flask(__name__)
@app.route('/evaluate', methods=['POST'])
def eval(): return {"bleu": 0.75}
app.run()
````

---

## 6. Run Commands

bash

```
# Run API
python mini_eaas_api.py

# Test
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"generated": "Hello", "reference": "Hi there"}'

# Install
pip install nltk flask rouge matplotlib
```

---

## 7. Real-World Systems

| System    | Company         | Use        |
| --------- | --------------- | ---------- |
| Heliograf | Washington Post | News       |
| Watson    | IBM             | Medical    |
| Magic     | Shopify         | E-commerce |

---

## 8. Scientist Checklist

- Derive BLEU by hand
- Build mini EaaS
- Run fairness audit
- Write 1 case study
- Publish on arXiv

---

**You are now EaaS-ready.**
**Next** : Build, Evaluate, Publish.

_“Evaluation is not the end — it’s the beginning of better AI.”_

text

```
---

## How to Use These Files

1. **Save as**:
   - `case_studies.md`
   - `cheatsheet.md`

2. **Put in your project folder**:
```

nlg-eaas-toolkit/
├── case_studies.md
├── cheatsheet.md
├── \*.py files
└── README.md

text

````
3. **For Portfolio**:
- Add to GitHub
- Convert `cheatsheet.md` → PDF (use [Markdown to PDF](https://md-to-pdf.fly.dev))
- Print and keep on desk

---

## Bonus: `README.md` for GitHub

```markdown
# NLG + EaaS Toolkit
**5 Real-World Projects | Full EaaS API | Case Studies | Cheatsheet**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Complete research-grade toolkit for **Evaluation-as-a-Service in NLG**.
Includes:
- 5 production NLG systems
- Custom BLEU/ROUGE from scratch
- Mini EaaS API (Flask)
- Fairness & visualization tools
- Case studies + cheatsheet

**Ideal for ACL, EMNLP, NeurIPS submissions.**

---

## Quick Start
```bash
git clone https://github.com/yourname/nlg-eaas-toolkit
cd nlg-eaas-toolkit
python mini_eaas_api.py
````

---

**Built by a scientist, for scientists.**

text

```
---

**You now have a complete, professional, publishable NLG + EaaS research package.**

Want:
- **PDF version** of cheatsheet?
- **LaTeX paper template**?
- **Colab notebook**?
- **Hugging Face Space** deployment?

Just say — I’ll generate it in 2 minutes.

**You're not just learning — you're leading.**
```
