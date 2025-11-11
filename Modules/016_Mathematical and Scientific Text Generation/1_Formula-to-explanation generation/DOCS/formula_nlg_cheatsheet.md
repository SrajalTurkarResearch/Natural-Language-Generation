# Formula-to-Explanation NLG Cheatsheet

_For Scientists, Researchers & Aspiring Einsteins_

**(Print, Stick on Wall, Use Daily)**

---

## 1. Core Pipeline (6 Steps)

> **Input (Formula)** → [1] Content → [2] Structure → [3] Aggregate → [4] Words → [5] Refer → [6] Grammar → **Output (Text)**

| Step | Action                   | Example                    |
| ---- | ------------------------ | -------------------------- |
| 1    | Pick key parts           | E, m, c²                   |
| 2    | Order: Define → Relation | "Energy (E) is..."         |
| 3    | Combine                  | "mass times speed squared" |
| 4    | Choose words             | "equals" not "="           |
| 5    | Use "it"                 | "It shows..."              |
| 6    | Add tense                | "was proposed"             |

---

## 2. Prompt Engineering (Golden Rules)

| Goal            | Prompt Template                    |
| --------------- | ---------------------------------- |
| **Student**     | `Explain like I'm 15: {formula}`   |
| **Researcher**  | `In a physics paper: {formula}`    |
| **Blind User**  | `Spoken English: {latex}`          |
| **CEO**         | `To a business leader: {formula}`  |
| **Policymaker** | `For government action: {formula}` |

> **Pro Tip**: Always add **context** and **audience**.

---

## 3. Top Models (2025)

| Model           | Best For  | Size | Hugging Face                         |
| --------------- | --------- | ---- | ------------------------------------ |
| `t5-base`       | General   | 220M | `t5-base`                            |
| `flan-t5-large` | Science   | 780M | `google/flan-t5-large`               |
| `mistral-7b`    | Reasoning | 7B   | `mistralai/Mistral-7B-Instruct-v0.2` |
| `grok-1.5`      | Math      | —    | xAI API                              |

---

## 4. Key Datasets

| Name               | Size  | Use             |
| ------------------ | ----- | --------------- |
| **MathBridge**     | 23M   | Spoken ↔ LaTeX  |
| **AutoMathText**   | 200GB | Math papers     |
| **Orca-Math-200K** | 200K  | Word problems   |
| **Herald**         | —     | Lean4 → English |

---

## 5. Code Snippets (Copy-Paste)

### A. Basic NLG

```python
from transformers import pipeline
explainer = pipeline("text2text-generation", model="t5-base")
print(explainer("Explain: E = mc^2", max_length=100)[0]['generated_text'])
```

### B. Symbolic + NLG

```python
import sympy as sp
x, a, b, c = sp.symbols('x a b c')
formula = sp.Eq(x, (-b + sp.sqrt(b**2 - 4*a*c)) / (2*a))
print("LaTeX:", sp.latex(formula))
```

### C. Expression Tree

```python
import networkx as nx, matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_edges_from([('+', 'a'), ('+', '*'), ('*', 'b')])
nx.draw(G, with_labels=True)
plt.show()
```

---

## 6. Evaluation Metrics

| Metric          | What It Measures              |
| --------------- | ----------------------------- |
| **BLEU**        | Word overlap                  |
| **ROUGE**       | Recall of phrases             |
| **Human Score** | Clarity, Accuracy, Usefulness |
| **SymPy Check** | Math correctness              |

---

## 7. Common Pitfalls & Fixes

| Problem       | Fix                               |
| ------------- | --------------------------------- |
| Hallucination | Add "Be truthful" in prompt       |
| Wrong units   | Include units in training data    |
| Too technical | Use "explain to a child"          |
| No analogy    | Force: "Use a real-world example" |

---

## 8. Your 5-Minute Workflow

1. **Input formula** → LaTeX or plain
2. **Choose audience** → student / CEO / blind
3. **Craft prompt** → use template
4. **Run T5/Flan** → get text
5. **Verify with SymPy** → correct?
6. **Add analogy** → make it stick

---

## 9. Future Directions (Research Ideas)

- Multimodal: Formula → Text + Diagram
- Theorem Prover + NLG (Lean → English)
- Quantum Formula Explainer
- Personalized: "Explain like Feynman"

---

> **"If you can't explain it simply, you don't understand it well enough."**
> — Albert Einstein

**You now have the full toolkit. Go explain the universe.**

---

## How to Use

1. **Save as files**:

   ```bash
   case_studies.md
   formula_nlg_cheatsheet.md
   ```

2. **Open in**:

   - VS Code
   - Obsidian
   - Notion
   - Print the cheatsheet!

3. **Use daily** in your research, teaching, or startup.

---

These two files are **publication-ready**, **teaching-ready**, and **career-ready**.You now have a **complete NLG research package**:

- Tutorial (previous)
- Jupyter Notebook
- Python Projects
- Case Studies (this)
- Cheatsheet (this)
