# Case Studies on Execution-Based Evaluation in Natural Language Generation (NLG)

This document complements the Jupyter Notebook `Execution_Based_Evaluation_NLG_Tutorial.ipynb` and the Python files (`setup_database.py`, `evaluate_nlg.py`, `visualize_metrics.py`, `mini_project.py`) by providing in-depth case studies on execution-based evaluation in NLG. Each case study is crafted for aspiring scientists, offering context, methodology, impact, lessons, challenges, and connections to your research path. Inspired by Marie Curie’s experimental rigor, Alan Turing’s computational precision, and Richard Feynman’s clear storytelling, these cases show how execution-based evaluation drives real-world impact and opens research opportunities.

## Case Study 1: Spider Dataset for Text-to-SQL (Yale, 2018)

### Context

The Spider dataset is a benchmark for testing NLG systems that convert natural language questions into SQL queries. It includes 10,000 questions across 200 diverse databases (e.g., university course schedules, hospital patient records). For example, a question like “How many students are enrolled in Biology?” becomes `SELECT COUNT(*) FROM courses WHERE subject='Biology';`. Execution-based evaluation tests if these queries produce correct results when run.

### Methodology

- **NLG Task** : A model (e.g., T5 from the `transformers` library) generates SQL from questions.
- **Evaluation** : Queries are executed on SQLite databases (like in `setup_database.py`). Results are compared to ground-truth answers using Execution Accuracy (EX) and Test-Suite Accuracy (TS), as implemented in `evaluate_nlg.py`.
- **Metrics** : EX measures % of queries returning correct results (e.g., 75% for T5). TS averages performance across test cases.
- **Example** : Question: “List departments with more than 5 employees.” Query: `SELECT department FROM employees GROUP BY department HAVING COUNT(*) > 5;`. Run using `evaluate_nlg.py` to check results.

### Impact

- **Business** : Enables non-technical users to query data (e.g., Amazon analyzing sales trends), saving time and reducing errors.
- **Research** : Standardizes NLG evaluation, allowing fair comparisons across models.

### Lessons Learned

- Models excel on simple queries but struggle with complex ones (e.g., nested subqueries or multi-table joins).
- Generalization to unseen databases is a challenge, highlighting a research gap.

### Challenges

- **Schema Variability** : Databases vary in structure, requiring robust NLG.
- **Error Handling** : Invalid queries (e.g., syntax errors) need detection, as shown in `evaluate_nlg.py`.

### Scientific Connection

Like Newton testing gravity across different objects, Spider tests NLG across diverse schemas. Use `mini_project.py` to simulate Spider-like tasks and explore generalization. **Research Idea** : Develop NLG models that adapt to new database schemas dynamically.

## Case Study 2: Virtual Assistants (e.g., Google Assistant)

### Context

Virtual assistants like Google Assistant use NLG to turn user questions (e.g., “What’s the stock price of Tesla?”) into executable API calls or queries. Execution-based evaluation ensures these calls return correct data, similar to SQL evaluation in `evaluate_nlg.py`.

### Methodology

- **NLG Task** : Generate API calls or database queries from questions.
- **Evaluation** : Run calls in a test environment (like a sandbox API). Compare results to expected outputs (e.g., correct stock price). Metrics include EX and F1 Score, calculable with `evaluate_nlg.py`.
- **Example** : Question: “Weather in Paris?” generates an API call to a weather service. Result (20°C) is checked for accuracy.

### Impact

- **User Experience** : Accurate responses (EX improved from 50% in 2016 to 80% in 2023) build trust.
- **Applications** : Used in smart homes, healthcare (e.g., querying patient data), and more.

### Lessons Learned

- Ambiguous questions (e.g., “What’s the time?” without location) cause errors.
- Execution-based eval catches functional failures that intrinsic metrics (e.g., BLEU) miss.

### Challenges

- **Ambiguity** : NLG must handle vague inputs.
- **Real-Time Constraints** : Queries must be fast, testable with Valid Efficiency Score (VES) in `evaluate_nlg.py`.

### Scientific Connection

Like Curie ensuring precise lab measurements, execution-based eval ensures reliability in critical applications. Visualize performance trends with `visualize_metrics.py`. **Research Idea** : Improve NLG for ambiguous inputs using context-aware models.

## Case Study 3: GitHub Copilot – Code Generation

### Context

GitHub Copilot generates code from natural language comments (e.g., “Write a function to sum numbers 1 to 10”). Execution-based evaluation runs the code to verify correctness, similar to SQL testing in `mini_project.py`.

### Methodology

- **NLG Task** : Generate Python or other code from comments.
- **Evaluation** : Run code in a sandbox (e.g., Python interpreter) and compare outputs to expected results, using EX and F1 metrics (adaptable from `evaluate_nlg.py`). Datasets like HumanEval test ~160 coding tasks.
- **Example** : Comment: “Sum 1 to 10.” Code: `sum(range(1, 11))`. Run to check if output is 55.

### Impact

- **Productivity** : Speeds up coding by 30-50% (GitHub studies).
- **Education** : Helps students learn programming by generating examples.

### Lessons Learned

- Syntax correctness doesn’t guarantee functional correctness (e.g., wrong logic).
- Execution-based eval is critical for catching logical errors.

### Challenges

- **Complexity** : Advanced tasks (e.g., recursive algorithms) are harder.
- **Safety** : Must prevent harmful code execution (e.g., deleting files).

### Scientific Connection

Like Turing’s work on computability, this tests if NLG produces functional outputs. Extend `evaluate_nlg.py` for code evaluation. **Research Idea** : Evaluate NLG for niche languages or scientific computing tasks.

## Case Study 4: Scientific Data Querying (e.g., CERN)

### Context

At CERN, NLG generates SQL queries to analyze physics data (e.g., particle collision speeds). Execution-based evaluation ensures queries return accurate results, critical for scientific discovery.

### Methodology

- **NLG Task** : Generate SQL from questions like “What’s the average particle speed in collisions?”
- **Evaluation** : Run queries on test databases (like `setup_database.py`). Check EX and TS metrics using `evaluate_nlg.py`.
- **Example** : Query: `SELECT AVG(speed) FROM collisions WHERE experiment='LHC';`. Verify result matches true average.

### Impact

- **Science** : Automates data analysis, saving researchers time.
- **Accuracy** : Ensures reliable results for publications.

### Lessons Learned

- Domain-specific terms (e.g., “quark”) challenge NLG models.
- Complex queries (e.g., multi-table joins) require robust evaluation.

### Challenges

- **Domain Knowledge** : NLG needs physics-specific training.
- **Scalability** : Large datasets require efficient queries (test with VES).

### Scientific Connection

Like Franklin’s precise DNA imaging, accurate queries are vital. Use `mini_project.py` to simulate CERN-like tasks. **Research Idea** : Develop NLG for interdisciplinary scientific queries (e.g., biology, chemistry).

## How to Use These Case Studies

- **Learn** : Understand how execution-based evaluation drives real-world impact.
- **Experiment** : Use `mini_project.py` to test similar questions or adapt `evaluate_nlg.py` for new domains.
- **Visualize** : Plot results with `visualize_metrics.py` to analyze trends.
- **Research** : Tackle gaps (e.g., complex queries, ambiguity) for your projects.
- **Inspire** : Like Curie, use these to design experiments that push NLG boundaries.
