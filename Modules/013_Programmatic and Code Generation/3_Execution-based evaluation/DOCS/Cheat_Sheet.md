# Cheat Sheet: Execution-Based Evaluation in NLG

This cheat sheet summarizes the **Execution-Based Evaluation in Natural Language Generation (NLG)** tutorial from `Execution_Based_Evaluation_NLG_Tutorial.ipynb` and related `.py` files. Designed for aspiring scientists, it provides key concepts, formulas, code snippets, and tips in a concise, researcher-friendly format. Inspired by Feynman’s clarity, Turing’s logic, and Curie’s precision, use this as a quick reference while studying or experimenting.

## 1. Key Concepts

- **Natural Language Generation (NLG)** : Computers turn data (e.g., `{city: 'Paris', temp: 20}`) into human-like text (e.g., “It’s 20°C in Paris”).
- **Analogy** : Like a chef turning ingredients (data) into a recipe (text).
- **Subtasks** : Data-to-text, text-to-text, code generation (e.g., SQL).
- **Evaluation Types** :
- **Intrinsic** : Checks text quality (e.g., fluency, grammar).
- **Extrinsic** : Tests task usefulness (e.g., answering questions).
- **Execution-Based** : Runs NLG output (e.g., SQL, code) and checks results.
- **Execution-Based Evaluation** : Tests if NLG output (e.g., `SELECT COUNT(*) FROM employees`) produces correct results when executed.
- **Why Special?** : Ensures functionality, not just appearance (like Turing’s computability).
- **Use Cases** : Text-to-SQL, code generation, API calls.

## 2. Key Metrics

- **Execution Accuracy (EX)** :
- **What** : % of outputs that run and match true results.
- **Formula** : EX = (Correct / Total) × 100%
- **Example** : 7/10 correct queries → EX = 70%.
- **Valid Efficiency Score (VES)** :
- **What** : % of outputs that run correctly and fast.
- **Example** : 4/10 efficient queries → VES = 40%.
- **Test-Suite Accuracy (TS)** :
- **What** : Average correctness across test cases.
- **Formula** : TS = (Matches / Tests)
- **Example** : 4/5 tests correct → TS = 80%.
- **F1 Score** :
- **What** : Balances precision (correct outputs given) and recall (correct outputs not missed).
- **Formula** : F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Example** : TP=3, FP=2, FN=1 → Precision=3/5=0.6, Recall=3/4=0.75, F1≈0.667.

## 3. Key Code Snippets

- **Setup Database** (from `setup_database.py`):
  ```python
  import sqlite3
  conn = sqlite3.connect(':memory:')
  cursor = conn.cursor()
  cursor.execute('CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)')
  cursor.executemany('INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)', [
      ('Alice', 'Sales', 50000), ('Bob', 'Sales', 55000), ('Charlie', 'IT', 60000)
  ])
  conn.commit()
  ```
- **Evaluate Query** (from `evaluate_nlg.py`):
  ```python
  def evaluate_query(conn, sql_query, true_result):
      cursor = conn.cursor()
      try:
          result = cursor.execute(sql_query).fetchall()
          result = result[0][0] if len(result) == 1 and isinstance(true_result, int) else [r[0] for r in result]
          return result == true_result
      except sqlite3.Error:
          return False
  ```
- **Visualize Metrics** (from `visualize_metrics.py`):
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd
  data = pd.DataFrame({
      'Query': ['Q1', 'Q2', 'Q3'] * 2,
      'Score': [70, 80, 60] + [65, 75, 55],
      'Metric': ['EX'] * 3 + ['F1'] * 3
  })
  sns.barplot(x='Query', y='Score', hue='Metric', data=data)
  plt.show()
  ```

## 4. Key Tips for Scientists

- **Setup** : Use `setup_database.py` to create test databases. Add more tables for complexity.
- **Evaluate** : Run `evaluate_nlg.py` to test queries. Add metrics like VES for efficiency.
- **Visualize** : Use `visualize_metrics.py` to plot results. Try real data from Spider.
- **Experiment** : Run `mini_project.py` for hands-on practice. Extend with real NLG models (e.g., T5).
- **Research** : Tackle gaps like complex query generation or generalization to new schemas.
- **Ethics** : Ensure safe execution (no harmful queries, like deleting data).

## 5. Common Pitfalls

- **Mistake** : Ignoring syntax errors in queries. **Fix** : Use try-except blocks (see `evaluate_nlg.py`).
- **Mistake** : Testing on changing databases. **Fix** : Use fixed test databases (like `:memory:`).
- **Mistake** : Focusing only on EX. **Fix** : Combine with F1, VES for robustness.

## 6. Research Ideas

- Develop NLG models for unseen database schemas.
- Combine execution-based and human evaluations for better accuracy.
- Explore NLG for scientific queries (e.g., physics, biology).
- Test NLG for new domains like quantum computing instructions.

## 7. Quick Start

1. Install: `pip install pandas matplotlib seaborn`.
2. Run `setup_database.py` to create a test database.
3. Use `evaluate_nlg.py` to test queries and calculate metrics.
4. Visualize with `visualize_metrics.py`.
5. Try `mini_project.py` for a hands-on project.
6. Read `Case_Studies.md` for real-world inspiration.

## 8. Why This Matters

Execution-based evaluation ensures NLG outputs _work_ , not just look good—like Curie’s precise lab results or Turing’s functional machines. Use this to build reliable AI for your research.
