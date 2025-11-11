# Cheatsheet: Logic-to-Text Mapping in NLG

This cheatsheet summarizes key concepts, steps, code, visualizations, and research ideas from the Logic-to-Text Mapping in NLG tutorial. It’s a quick reference for aspiring scientists, designed to be clear, beginner-friendly, and inspiring, like a notepad for Turing’s algorithms, Einstein’s equations, or Tesla’s blueprints. Use it alongside the Jupyter Notebook and `.py` files.

## 1. Key Concepts

- **NLG (Natural Language Generation)** : Computers turning facts into human-like sentences.
- **Analogy** : Like a chef making a dish (sentences) from ingredients (facts).
- **Subfields** : Data-to-text, text-to-text, image-to-text, **logic-to-text** .
- **Logic-to-Text Mapping** : Converting strict rules (logic) to sentences.
- **Why Cool?** : Control (edit rules), truth (no made-up facts), explainable (show rules).
- **Example** : Logic: `Eats(John, Apple)` → Text: “John eats an apple.”
- **Logic Types** :
- **Propositional** : True/false (e.g., `P → Q` = “If raining, take umbrella”).
- **Predicate** : Things and links (e.g., `∀x (Human(x) → Mortal(x))` = “All humans are mortal”).
- **Lambda** : Functions (e.g., `λx. Eats(John, x)` = “John eats something”).
- **Description** : Knowledge maps (e.g., Bird is Animal with wings).
- **Modal** : Possibility/must (e.g., `◇Rains` = “Might rain”).
- **Math Insight** : Truth table for `P ∧ Q`:

```
  P | Q | P ∧ Q
  T | T | T
  T | F | F
  F | T | F
  F | F | F
```

## 2. Mapping Process (8 Steps)

1. **Parse Logic** : Check rules are correct.
2. **Content Selection** : Pick key facts (score: frequency × importance).
3. **Discourse Planning** : Order ideas (e.g., cause then effect).
4. **Lexicalization** : Choose words (e.g., “Likes” → “enjoys”).
5. **Aggregation** : Combine facts (e.g., “John eats apple AND likes fruit”).
6. **Referring Expressions** : Use “he” instead of “John” again.
7. **Surface Realization** : Add grammar (e.g., “is” vs. “are”).
8. **Evaluation** : Check with BLEU (word matches) or METEOR (similar words).

**Visualization** : Flowchart:

```
[Logic] → Parse → Select → Plan → Words → Combine → Pronouns → Grammar → [Text] → Check
```

## 3. Code Snippets

- **Simple Mapper** (`simple_logic_mapper.py`):
  ```python
  def simple_logic_to_text(logic):
      if logic['type'] == 'Predicate':
          return f"{logic['subject']} {logic['verb']} {logic['object']}."
  # Example: {'type': 'Predicate', 'subject': 'John', 'verb': 'eats', 'object': 'apple'} → "John eats apple."
  ```
- **Advanced Mapper** (`advanced_logic_mapper.py`):
  ```python
  def advanced_logic_to_text(logic):
      if logic['type'] == 'Predicate':
          return f"{logic['subject']} {logic['verb']} {logic['object']}."
      elif logic['type'] == 'AND':
          return f"{advanced_logic_to_text(logic['left'])[:-1]} and {advanced_logic_to_text(logic['right']).lower()}"
  # Example: AND(Eats(John, Apple), Likes(John, Fruit)) → "John eats apple and likes fruit."
  ```
- **Visualization** (`visualization_dataset.py`):
  ```python
  from graphviz import Digraph
  dot = Digraph()
  dot.node('A', 'AND')
  dot.node('B', 'Eats(John, Apple)')
  dot.node('C', 'Likes(John, Fruit)')
  dot.edges(['AB', 'AC'])
  dot.render('logic_tree', format='png')
  ```

## 4. Visualizations

- **Logic Tree** : For `AND(Eats(John, Apple), Likes(John, Fruit))`:

```
  AND
  ├── Eats(John, Apple)
  └── Likes(John, Fruit)
```

- **Flowchart** : See mapping process above (use Matplotlib to draw).

## 5. Applications

- **Healthcare** : Logic → Patient reports (e.g., “John has fever, see doctor”).
- **Sports** : Stats → Game recaps (e.g., “LeBron scored 30 points”).
- **Science** : Data → Summaries (e.g., “Global temp rose 1.2°C”).
- **Education** : Rules → Explanations (e.g., “Triangle is right because 90°”).
- **See Details** : `Case_Studies_NLG.md`.

## 6. Datasets

- **Logic2Text** : 10,000 table-logic-text examples (Hugging Face).
- **LogiQA 2.0** : Logic puzzles.
- **ROTWIRE** : Sports stats.
- **WikiTableQuestions** : Table queries.
- **Usage** : `from datasets import load_dataset; dataset = load_dataset('logic2text')`

## 7. Research Ideas

- **Ethics** : Detect bias in logic (e.g., gendered predicates).
- **Hybrid Systems** : Combine logic with LLMs like GPT.
- **Multimodal** : Map logic from images/videos.
- **Quantum NLG** : For probabilistic logics.
- **Green NLG** : Reduce energy use in models.

## 8. Exercises

1. Map `If Rain, then Umbrella` to text. Draw tree.
   - **Answer** : “If it rains, take an umbrella.” Tree: `If → Rain → Umbrella`.
2. Code mapper for `Owns(Bob, Car)` → “Bob owns a car.”
3. Try one Logic2Text example.

## 9. Challenges

- **Hallucination** : Models add wrong facts. Fix: Tie to logic.
- **Scalability** : Hard for big logics. Fix: Smarter algorithms.
- **Multilingual** : Tough for non-English. Fix: Train on many languages.
- **Ethics** : Risk of fake news. Fix: Verify truth.

## 10. Next Steps

- **Learn** : Python NLTK (basics), Hugging Face Transformers (advanced).
- **Read** : Logic2Text paper (arxiv.org/abs/2004.14579).
- **Experiment** : Try ROTOWIRE dataset or neural T5 model.
- **Plan** : “By Nov 2025, map weather logic to text.”

## 11. What’s Missing in Other Tutorials

- Beginner-friendly code (this has simple Python).
- Ethics focus (we cover bias detection).
- Research prompts (we include “Research Lab”).
- Practical projects (we have mini/major projects).

  **Research Lab** : Pick one idea (e.g., ethical NLG). Write a one-sentence experiment plan.
