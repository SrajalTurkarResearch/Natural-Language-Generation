# Cheatsheet: Lambda Calculus, DRS, and AMR in NLG

**Quick Reference Guide** : This cheatsheet summarizes the tutorial on Lambda Calculus, Discourse Representation Structures (DRS), and Abstract Meaning Representation (AMR) for Natural Language Generation (NLG). Designed as a one-page (or quick-scan) resource for aspiring scientists like you, it condenses theory, key formulas, examples, code snippets, visualizations, applications, and 2025 updates. Use it alongside your notes and .py files—think of it as Turing's quick logic table, Einstein's equation summary, or Tesla's circuit sketch. Print, bookmark, or copy for your research toolkit.

## 1. NLG Overview

- **Definition** : Turn data into text: Content Determination → Sentence Planning → Surface Realization.
- **Semantics Role** : Lambda (functions), DRS (context), AMR (graphs) ensure logical meaning.
- **Analogy** : Data (raw power) → Tools (transformers) → Text (light bulb), like Tesla's AC.
- **Key Pitfall** : Without semantics, AI "hallucinates" wrong facts.
- **2025 Update** : Multimodal NLG (text + images) in AI assistants.

  **Visualization** : Pyramid Flowchart – Bottom: Data → Middle: Semantics → Top: Text.

## 2. Lambda Calculus

- **Basics** : Functions only. Symbols: λ (function), x (variable), (M N) (apply).
- **Rules** :
- Alpha: λx.M → λy.M (rename).
- Beta: (λx.M) N → M[x=N] (substitute).
- Eta: λx.(f x) → f.
- **Church Numerals** : 0 = λf.λx.x, 1 = λf.λx.f x, Add = λm.λn.λf.λx.m f (n f x).
- **Recursion** : Y = λf.(λx.f (x x)) (λx.f (x x)).
- **Example** : Factorial = Y (λf.λn. n==0 ? 1 : n\*f(n-1)).
- **Code Snippet** (from lambda_calculus.py):
  ```
  add = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))
  result = add(two)(three)(inc)(0)  # 5
  ```
- **NLG Application** : Compose meanings, e.g., λx.λy.love(x,y)(John)(Mary).
- **Pitfall** : Infinite reductions (Omega: (λx.x x)(λx.x x)).
- **2025 Update** : Neural Lambda for AI reasoning; Parallel Lambda for speed.

  **Visualization** : Reduction Tree – (λx.x+1)5 → 5+1 → 6.

## 3. DRS (Discourse Representation Structures)

- **Basics** : Boxes for meaning. Universe: [x,y] (variables). Conditions: man(x).
- **Operations** : Merge boxes; Accessibility for pronouns (e.g., "he" reaches back).
- **Quantifiers** : Every (∀) = ⇒ box; A (∃) = new variable.
- **Advanced** : Tense (e < now); Modals (♦ P(x)); Plurals (sum(x,y)).
- **Example** : "If man happy, he smiles": [] ⇒ [x] man(x) happy(x) [] smiles(x).
- **Code Snippet** (from drs.py):
  ```
  drs = DRS(['x'], ['man(x)', 'smiles(x)'])
  print(drs)  # Universe: ['x'] Conditions: ['man(x)', 'smiles(x)']
  ```
- **NLG Application** : Multi-sentence coherence, e.g., "Scientist invents. It works."
- **Pitfall** : Variable capture in nesting.
- **2025 Update** : Universal DRT (word-anchored); Graph DRS for multimodal.

  **Visualization** : Nested Boxes – Outer [] ⇒ Inner [x] conditions [] conditions.

## 4. AMR (Abstract Meaning Representation)

- **Basics** : Graphs. Nodes: Concepts (jump-01); Edges: Relations (:ARG0 agent).
- **Operations** : Parsing (text → graph); Smatch F1 = 2*(P*R)/(P+R).
- **Advanced** : Polarity (- for no); Reentrancy (loops for "he").
- **Example** : "Fox jumps dog": (jump-01 :ARG0 fox :ARG1 dog).
- **Code Snippet** (from amr.py):
  ```
  G = nx.DiGraph()
  G.add_edges_from([('jump-01', 'fox', {'label': ':ARG0'})])
  nx.draw(G, with_labels=True)
  plt.show()
  ```
- **NLG Application** : Abstract meanings for variations, e.g., "John eats apple."
- **Pitfall** : Overly complex graphs.
- **2025 Update** : Neural AMR parsing; AMR-DA for data augmentation.

  **Visualization** : Node-Link Diagram – Root → Edges to args/mods.

## 5. Integration in NLG Pipeline

- **Flow** : Data → AMR (abstract) → DRS (context) → Lambda (compute) → Text.
- **Example** : Physics data → AMR graph → DRS box → Lambda force → "Object accelerates..."
- **Code Snippet** (from nlg_project.py): Combines all for report generation.
- **2025 Update** : Neurosymbolic hybrids (symbolic tools + LLMs).

## 6. Quick Tips for Scientists

- **Turing Approach** : Prove correctness with reductions.
- **Einstein Approach** : Unify abstractions for elegant models.
- **Tesla Approach** : Experiment iteratively with code/projects.
- **Exercises** : Lambda: Reduce (λx.λy.y x) a b = b a; DRS: "No one knows everything" = ¬[x] person(x) [y] thing(y) knows(x,y); AMR: Parse "E=mc²".
- **Research Directions** : Quantum NLG (lambda qubits); Ethical AI (DRS coherence); Multilingual (AMR graphs).
- **What's Missing in Standard Tutorials** : Integration pipelines, 2025 updates, scientific applications—covered here!

  **Final Note** : Use this cheatsheet daily; cross-reference with .py files and case studies. Innovate like the greats—your discoveries start here!
