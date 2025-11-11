# Case Studies in Human-Centered Evaluation of Natural Language Generation (NLG)

> **For Aspiring Scientists & Researchers** > _“Science is not just theory — it is tested in the real world.”_
> — Inspired by Einstein, Turing, and Tesla

---

## Case Study 1: Medical Text Simplification Using HCRS

**Source**: _HCRS: A Human-Centered Framework for Health Text Readability (2025)_

### Problem

Patients receive complex medical reports (e.g., "LVEF 35%") and cannot act on them.

### NLG Task

Convert clinical notes → plain language summaries.

### Evaluation Method

**HCRS (Human-Centered Readability Score)**4 Dimensions (1–5 scale):

- **Clarity**: Is it easy to read?
- **Trustworthiness**: Does it feel accurate?
- **Actionability**: Can the patient act on it?
- **Engagement**: Is it interesting?

### Data

- 50 real discharge summaries (anonymized)
- 200 patients + 50 clinicians

### Results

| Dimension       | Before (Raw) | After (Simplified) | Δ     |
| --------------- | ------------ | ------------------ | ----- |
| Clarity         | 2.1          | 4.6                | +119% |
| Trustworthiness | 3.8          | 4.4                | +16%  |
| Actionability   | 1.9          | 4.2                | +121% |
| Engagement      | 2.5          | 3.8                | +52%  |

### Key Insight

> **ROUGE score dropped 12% but patient comprehension rose 45%**
> → **Traditional metrics fail in high-stakes domains.**

### Research Direction

Integrate **eye-tracking** to measure cognitive load during reading.

---

## Case Study 2: Customer Service Chatbot — LLM + Human Hybrid Eval

**Source**: _EMNLP 2025 Industry Track_

### Problem

Chatbots resolve 60% of queries but frustrate users with robotic tone.

### NLG Task

Generate empathetic, accurate responses.

### Evaluation

1. **LLM-as-Judge** (GPT-4): Scores helpfulness (1–5)
2. **Human Expert Override**: Flags empathy gaps
3. **Customer Follow-up**: “Did this resolve your issue?”

### Data

- 1,000 real customer chats
- 3 versions: Direct, Polite, Empathetic

### Results

| Response Type | LLM Score | Human Override | Resolution Rate |
| ------------- | --------- | -------------- | --------------- |
| Direct        | 4.2       | 62% rejected   | 68%             |
| Polite        | 4.5       | 18% rejected   | 81%             |
| Empathetic    | 4.1       | 5% rejected    | **92%**         |

### Key Insight

> **Empathy > Fluency** in user satisfaction.
> LLM overrates brevity; humans value emotional connection.

### Research Direction

Train **Empathy-Aware LLMs** using think-aloud + sentiment data.

---

## Case Study 3: Automated News Summarization with Error Annotation

**Source**: _GEM Workshop @ INLG 2025_

### Problem

AI summaries hallucinate facts (e.g., “Tesla opened factory” → actually announced).

### NLG Task

Article → 3-sentence summary.

### Evaluation

**Expert Error Taxonomy** (5 types):

1. Factual Error
2. Entity Error
3. Magnitude Error
4. Temporal Error
5. Omission

### Data

- 500 CNN articles
- 3 NLG systems (T5, BART, GPT-4)

### Results

| System | Error Rate | Critical Errors | Human Trust Score |
| ------ | ---------- | --------------- | ----------------- |
| T5     | 28%        | 14              | 2.8               |
| BART   | 22%        | 9               | 3.5               |
| GPT-4  | 15%        | 3               | **4.6**           |

### Key Insight

> **Even 85% factual accuracy → 40% user distrust**
> → One critical hallucination destroys credibility.

### Research Direction

**Retrieval-Augmented Generation (RAG)** + **post-hoc fact-check layer**.

---

## Case Study 4: Educational NLG with Think-Aloud HCI

**Source**: _CHI 2026 Education Track (Accepted)_

### Problem

Students skim AI-generated explanations and retain nothing.

### NLG Task

Concept → engaging, analogy-rich explanation.

### Evaluation

**Think-Aloud Protocol** (10 middle-school students):

- Read aloud
- Say thoughts
- Recall test after 10 mins

### Data

| Version      | Metaphor Used  | Confusion Points | Recall Score |
| ------------ | -------------- | ---------------- | ------------ |
| Plain        | None           | 6                | 42%          |
| Analogy-Rich | “Like cooking” | 2                | **88%**      |

### Key Insight

> **Metaphors reduce cognitive load by 60%.**
> Think-aloud reveals _where_ understanding breaks.

### Research Direction

Build **adaptive NLG** that inserts analogies based on user confusion signals.

---

## Summary Table: Evaluation Methods Across Domains

| Domain     | Core Metric          | Human Involvement     | Best For                  |
| ---------- | -------------------- | --------------------- | ------------------------- |
| Healthcare | HCRS                 | Patients + Clinicians | Actionability, Trust      |
| Chatbots   | LLM + Human Veto     | Support Agents        | Empathy, Resolution       |
| News       | Error Taxonomy       | Fact-Checkers         | Factual Accuracy          |
| Education  | Think-Aloud + Recall | Students + Teachers   | Comprehension, Engagement |

---

> **Your Research Prompt**:
> Pick **one case**, replicate it with real data, improve the evaluation, and submit to **INLG 2026**.

---
