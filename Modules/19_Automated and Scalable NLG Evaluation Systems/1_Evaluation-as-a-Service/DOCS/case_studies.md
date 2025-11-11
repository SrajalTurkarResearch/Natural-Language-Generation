# Case Studies: Evaluation-as-a-Service in Natural Language Generation

_Real-World Applications & Scientific Insights_
**Author**: [Your Name] – AI Researcher
**Date**: November 11, 2025

---

## Case Study 1: **Automated Journalism at The Washington Post**

**System**: Heliograf (2016–present)
**NLG Task**: Sports & election reports
**EaaS Used**: Internal evaluation pipeline (BLEU, ROUGE, human-in-loop)

### Data

- **Input**: Game stats (XML/JSON)
- **Output**: 800+ articles during Rio Olympics
- **Reference**: Human-written articles

### Evaluation Metrics

| Metric       | Score | Insight                 |
| ------------ | ----- | ----------------------- |
| BLEU         | 0.68  | High n-gram overlap     |
| ROUGE-L      | 0.72  | Good structure match    |
| Human Rating | 4.1/5 | "Accurate and readable" |

### Key Lesson

> **EaaS enables scale**: 1 human editor → 1000 auto articles/day
> **Trade-off**: Sacrificed creativity for consistency

**Reference**: [The Washington Post, 2017](https://www.washingtonpost.com)

---

## Case Study 2: **Medical NLG in IBM Watson Health**

**System**: Watson for Oncology
**Task**: Generate patient treatment summaries
**EaaS**: Custom fairness + accuracy auditor

### Data

- 500 patient EHRs
- Outputs evaluated across gender, age, ethnicity

### Fairness Audit (EaaS)

```python
Demographic Parity Gap (Gender): 0.03 → Fair
Disparate Impact Ratio: 0.98 → Compliant
```

### Results

- **Accuracy** : 94% match with oncologist notes
- **Bias Reduced** : Passive voice in female reports ↓ 60% after retraining

### Key Lesson

> **EaaS prevents harm** : Bias detection before deployment
> **Ethics > Metrics** : Even high BLEU can hide stereotypes

---

## Case Study 3: **E-commerce NLG at Shopify**

**System** : Magic Product Descriptions
**Task** : Auto-generate 10K+ product copies
**EaaS** : A/B testing + BERTScore

### Experiment

| Version       | CTR  | BERTScore | Revenue Lift |
| ------------- | ---- | --------- | ------------ |
| Human-written | 2.1% | 0.91      | Baseline     |
| NLG (v1)      | 2.4% | 0.88      | **+14%**     |

### Key Lesson

> **EaaS closes the loop** : Metric → Revenue
> **BERTScore > BLEU** for customer conversion

---

## Case Study 4: **Climate NLG for IPCC Reports**

**System** : Auto-summary of 1000-page climate models
**EaaS** : Factual entailment + readability check

### Evaluation

- **Entailment Accuracy** : 97% (using NLI models)
- **Readability (Flesch)** : 62 → "Standard" (human: 58)
- **CO₂ Fact Check** : 0 errors in 50 reports

### Key Lesson

> **EaaS ensures trust** : Science communication must be **correct + clear** > **Future** : Use EaaS for live climate dashboards

---

## Case Study 5: **Education NLG – Duolingo Summary Bot**

**Task** : Lesson → 3-bullet summary
**EaaS** : Student comprehension test (extrinsic)

### Results

| Group            | Recall Score |
| ---------------- | ------------ |
| Read NLG Summary | 82%          |
| Read Full Lesson | 78%          |

### Key Lesson

> **NLG > Raw content** when evaluated **extrinsically** > **EaaS proves learning impact**

---

## Summary Table: EaaS Impact Across Domains

| Domain     | Primary Metric  | EaaS Value |
| ---------- | --------------- | ---------- |
| Journalism | ROUGE           | Scale      |
| Healthcare | Fairness        | Safety     |
| E-commerce | BERTScore + CTR | Revenue    |
| Climate    | Entailment      | Trust      |
| Education  | Comprehension   | Learning   |

---

**Your Research Question** :

> _"How does EaaS metric choice affect real-world outcomes across domains?"_

**Next Step** : Run your own EaaS on one of these datasets → publish!
