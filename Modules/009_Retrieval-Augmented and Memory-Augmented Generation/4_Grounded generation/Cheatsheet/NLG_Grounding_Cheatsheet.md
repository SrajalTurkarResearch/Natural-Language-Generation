# NLG Grounding Cheatsheet: Quick Reference for Aspiring Scientists

This cheatsheet summarizes the **Grounded Generation in NLG** tutorial for quick reference. It links to the Python files (`theory.py`, `code_guides.py`, etc.) and is designed for your research notes. Use it to revise, plan experiments, or spark ideas.

---

## 1. Core Concepts

- **What is NLG?** : Computers writing human-like text from data (e.g., numbers to "Sunny day at 25°C"). See `theory.py` (#1).
- **Grounding** : Tying text to real facts (e.g., books, images) to avoid errors. Analogy: A map to stay on track.
- **Key Principles** :
- **Faithfulness** : Text matches facts. Metric: FactScore (matching words).
- **Relevance** : Use only needed facts. Metric: Cosine similarity.
- **Coherence** : Smooth, not robotic text.
- **Types** :
- Fact-Based (e.g., Wikipedia).
- Picture-Based (e.g., photo captions).
- Number-Based (e.g., table summaries).
- Mixed (text + images). See `theory.py` (#3).

---

## 2. Key Methods

- **Retrieval-Augmented Generation (RAG)** :

1. Turn question into numbers (embedding).
2. Find best facts (cosine similarity).
3. Write answer with facts. See `code_guides.py` (2.2).

- **Others** :
- Copy-Point: Mix new words and fact words.
- Knowledge Graphs: Web of facts (e.g., Paris → capital → France).
- Agentic Systems: AI teams (finder, writer). See `theory.py` (#5).

---

## 3. Code Snippets

- **Template NLG** (`code_guides.py`):
  ```python
  def template_nlg(fact_dict):
      return f"The fact is: {fact_dict.get('fact', 'unknown')}"
  print(template_nlg({'fact': 'Paris is France’s capital.'}))
  ```
- **RAG Simulation** (`code_guides.py`):
  ```python
  def rag_nlg(question, facts):
      q_embed = np.array([0.8, 0.6])
      best_score = -1
      best_fact = None
      for fact in facts:
          score = cosine_similarity(q_embed, fact['embedding'])
          if score > best_score:
              best_score = score
              best_fact = fact['text']
      return f"Based on facts: {best_fact}"
  ```
- **Mini Project** (`mini_project.py`):
  ```python
  def weather_nlg(city):
      for data in weather_data:
          if data['city'].lower() == city.lower():
              return f"In {data['city']}, it's {data['condition']} with {data['temp']}°C."
  ```

---

## 4. Visualizations

- **Accuracy Plot** (`visualizations.py`):
  - Line graph: More facts (x-axis) → Higher accuracy (y-axis, up to 95%).
  - Run: `plt.plot(facts_used, accuracy)`.
- **RAG Flow Diagram** (Sketch from `visualizations.py`):
  - Boxes: Question → Find Facts → Combine → Write Answer.
  - Dashed Path: No facts → Wrong answer.
  - Analogy: Librarian finding a book before summarizing.

---

## 5. Applications

- **Healthcare** : Summarize EHRs (e.g., "Normal BP: 120/80"). See `Detailed_Case_Studies.md` (Case 1).
- **Climate** : Data to reports (e.g., "Arctic warming +1.2°C"). See `mini_project.py`.
- **Education** : Engaging science posts. See `Detailed_Case_Studies.md` (Case 3).
- **Journalism** : Fact-checked news. See `major_project.py`.

---

## 6. Research Tips

- **2025 Trends** : Agentic AI, multimodal grounding, persuasive NLG. See `theory.py` (#5).
- **Rare Insight** : Bias in facts is a trap – audit sources. Grounding cuts errors by 20-50% (TruthfulQA, 2025).
- **Experiment Idea** : Adapt `major_project.py` for a new dataset (e.g., Wikipedia). Test: Does RAG beat ungrounded?

---

## 7. Exercises

- **Template NLG** (`exercises.py`): Write a new format (e.g., "Fact: [fact]").
- **Cosine Similarity** (`exercises.py`):
  ```python
  a = np.array([0.8, 0.6])
  b = np.array([0.9, 0.5])
  print(cosine_similarity(a, b))  # ~1.0
  ```
- **Task** : Add 5 facts to `major_project.py` and test.

---

## 8. What’s Missing in Other Tutorials

- **Science Mindset** : Asking “Why?” and “What if?”.
- **Math Clarity** : Full derivations (e.g., cosine in `exercises.py`).
- **Ethics** : Check fact bias. See `theory.py` (#6).
- **Real-World Links** : Cases like `Detailed_Case_Studies.md`.

---

## 9. Next Steps

- **Learn** : Study transformers (Hugging Face), retrieval (FAISS). See `code_guides.py` (2.3).
- **Experiment** : Build a chatbot with Wikipedia data.
- **Research** : Publish on grounding in your field (e.g., biology).
- **Reflect** : What NLG problem can you solve?

---

## Quick Links

- **Theory** : `theory.py`
- **Code** : `code_guides.py`, `mini_project.py`, `major_project.py`
- **Visuals** : `visualizations.py`
- **Practice** : `exercises.py`
- **Cases** : `Detailed_Case_Studies.md`

Use this cheatsheet to revise or plan experiments. Keep it handy in your research journal!
