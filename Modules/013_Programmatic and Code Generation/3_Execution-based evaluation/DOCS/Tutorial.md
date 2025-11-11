# Execution-Based Evaluation in Natural Language Generation: A Comprehensive Guide for Aspiring Scientists

Welcome, future scientist! This book is your ultimate guide to mastering **Execution-Based Evaluation in Natural Language Generation (NLG)** , crafted to take you from a curious beginner to a confident researcher. Inspired by the brilliance of Albert Einstein (simplifying complex ideas through thought experiments), Richard Feynman (explaining like a story), Isaac Newton (building knowledge step-by-step), Marie Curie (testing with precision), Alan Turing (designing logical systems), Ada Lovelace (envisioning computational creativity), and modern AI pioneers like Yoshua Bengio (advancing AI through empirical rigor), this tutorial is designed to be your one-stop resource. Since you’re relying solely on this to become an expert, I’ve made it comprehensive, detailed, and easy to understand, with no jargon, no hidden meanings, and every term explained as if it’s your first encounter.

This book is structured like a journey through a scientific landscape, with chapters that build your knowledge systematically. Each chapter includes clear explanations, analogies, math derivations, examples (beginner to advanced), visualizations, exercises, and research insights. It addresses gaps from standard tutorials by diving deeper into history, mathematics, edge cases, ethics, and interdisciplinary applications. You’ll also find connections to the Jupyter Notebook (`Execution_Based_Evaluation_NLG_Tutorial.ipynb`), Python files (`setup_database.py`, `evaluate_nlg.py`, `visualize_metrics.py`, `mini_project.py`), `Case_Studies.md`, and `Cheat_Sheet.md` for practical implementation.

Grab a notebook, take it slow, and let’s explore execution-based evaluation like scientists discovering a new universe!

## Table of Contents

1. [Introduction: Why NLG and Execution-Based Evaluation Matter](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-1-introduction-why-nlg-and-execution-based-evaluation-matter)
2. [Fundamentals of NLG: Building the Foundation](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-2-fundamentals-of-nlg-building-the-foundation)
3. [The Need for Evaluation: Ensuring Quality](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-3-the-need-for-evaluation-ensuring-quality)
4. [Execution-Based Evaluation: The Core Concept](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-4-execution-based-evaluation-the-core-concept)
5. [How It Works: Step-by-Step Process](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-5-how-it-works-step-by-step-process)
6. [Metrics and Mathematics: Measuring Success](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-6-metrics-and-mathematics-measuring-success)
7. [Practical Implementation: Code and Experiments](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-7-practical-implementation-code-and-experiments)
8. [Visualizing Results: Making Insights Clear](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-8-visualizing-results-making-insights-clear)
9. [Real-World Applications: Impact in Action](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-9-real-world-applications-impact-in-action)
10. [Research Directions: Pushing the Boundaries](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-10-research-directions-pushing-the-boundaries)
11. [Projects: Hands-On Learning](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-11-projects-hands-on-learning)
12. [Exercises: Build Your Skills](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-12-exercises-build-your-skills)
13. [What’s Missing in Standard Tutorials: Filling the Gaps](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-13-whats-missing-in-standard-tutorials-filling-the-gaps)
14. [Future Directions: Your Path Forward](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-14-future-directions-your-path-forward)
15. [Conclusion: Becoming an NLG Scientist](https://grok.com/c/5cbe2cce-342e-4e4e-a441-949e36f2f14a#chapter-15-conclusion-becoming-an-nlg-scientist)

---

## Chapter 1: Introduction – Why NLG and Execution-Based Evaluation Matter

### 1.1 What You’ll Learn

Natural Language Generation (NLG) is the art and science of teaching computers to create human-like text from raw data, like turning a spreadsheet into a story or a question into a database query. Execution-based evaluation is a powerful way to test NLG by running its outputs (e.g., SQL queries, code) and checking if they produce correct results. This book will teach you:

- **Fundamentals** : What NLG is, how it works, and its history.
- **Evaluation** : Why and how we test NLG, focusing on execution-based methods.
- **Practical Skills** : Coding, visualizing, and experimenting with NLG systems.
- **Research** : Cutting-edge ideas and gaps to explore.
- **Applications** : Real-world uses in business, science, and more.

### 1.2 Why It Matters

NLG is like a bridge between raw data and human understanding, enabling applications like automated reports, chatbots, and scientific data analysis. Execution-based evaluation ensures these outputs _work_ correctly, not just sound good—like Turing testing a computer’s logic or Curie verifying lab results. As a scientist, mastering this will let you automate tasks, validate AI, and publish groundbreaking research.

- **Feynman Analogy** : NLG is like a teacher explaining a complex experiment in simple words; execution-based evaluation checks if the explanation leads to the right answer.
- **Einstein Thought Experiment** : Imagine a world where computers write instructions for robots. How do you know the instructions are right? Run them and see!

### 1.3 How to Use This Book

- **Read Sequentially** : Start with fundamentals and build to advanced topics.
- **Practice** : Use the Python files (`setup_database.py`, etc.) to run code, referenced in Chapter 7.
- **Refer to Artifacts** : Check `Case_Studies.md` for real-world examples and `Cheat_Sheet.md` for quick summaries.
- **Take Notes** : Each chapter has clear sections, examples, and exercises for learning.
- **Think Like a Scientist** : Use research directions and projects to start your own experiments.

---

## Chapter 2: Fundamentals of NLG – Building the Foundation

### 2.1 What is NLG?

NLG is when a computer transforms data—numbers, tables, images, or text—into human-readable text, like sentences, reports, or even code. For example, given `{city: 'Paris', temp: 20}`, an NLG system might output “It’s 20 degrees Celsius in Paris today.”

- **Key Idea** : NLG makes data accessible by turning it into language.
- **Difference from NLU** : NLG _creates_ text; Natural Language Understanding (NLU) _interprets_ text.
- **Types of NLG** :
- **Data-to-Text** : Turns tables into narratives (e.g., weather reports).
- **Text-to-Text** : Summarizes or rephrases text (e.g., simplifying articles).
- **Code Generation** : Creates executable instructions (e.g., SQL queries, Python code).
- **Lovelace Analogy** : NLG is like a poet weaving stories from numbers, just as Lovelace saw computers creating beyond calculations.

### 2.2 How NLG Works

NLG systems follow a pipeline:

1. **Content Planning** : Decide what to say (e.g., focus on temperature for a weather report).
2. **Sentence Planning** : Choose words and structure (e.g., “It’s [temp] in [city]”).
3. **Realization** : Generate final text using rules or AI models.

Modern NLG uses **transformers** , neural networks trained on massive text datasets to predict the next word based on context. For example, given “It’s sunny in…”, a transformer might predict “Paris” based on input data.

### 2.3 History of NLG

- **1950s** : Alan Turing proposed machines could mimic human language (Turing Test).
- **1960s** : Eliza, an early chatbot, generated simple responses using templates.
- **1970s** : SHRDLU described virtual worlds (e.g., “The red block is on the table”).
- **1980s–2000s** : Statistical models learned patterns from text data.
- **2017** : Transformers (from “Attention is All You Need” paper) revolutionized NLG, enabling complex outputs like essays or SQL queries.
- **Why History Matters** : Like Darwin tracing species evolution, understanding NLG’s past helps you avoid outdated approaches (e.g., rigid templates) and build innovative systems.

### 2.4 Mathematics of NLG

NLG relies on **language models** that assign probabilities to word sequences. The probability of a sentence is the product of each word’s conditional probability:

- **Formula** : For a sentence ( W = w*1, w_2, \ldots, w_n ),
  [
  P(W) = P(w_1) \times P(w_2 | w_1) \times P(w_3 | w_1, w_2) \times \cdots \times P(w_n | w_1, \ldots, w*{n-1})
  ]
- **Example** : Sentence: “The cat sleeps.”
- Probabilities: ( P(\text{The}) = 0.7 ), ( P(\text{cat} | \text{The}) = 0.6 ), ( P(\text{sleeps} | \text{The, cat}) = 0.8 ).
- Total: ( 0.7 \times 0.6 \times 0.8 = 0.336 ).
- **Perplexity** : Measures how “surprising” a sentence is (lower is better):
  [
  \text{Perplexity} = 2^{-\frac{1}{n} \sum_{i=1}^n \log_2 P(w_i | w_1, \ldots, w_{i-1})}
  ]
- Calculation: For above, ( \log_2(0.336) \approx -1.573 ), ( -\frac{1}{3} \times -1.573 \approx 0.524 ), ( 2^{0.524} \approx 1.44 ).
- **Why Math?** : Like Curie measuring radiation, probabilities quantify NLG’s reliability.

### 2.5 Examples for All Levels

- **Beginner** : Input: `{animal: 'dog', action: 'runs'}` → Output: “The dog runs.”
- **Intermediate** : Input: `{sales: {Q1: 100k, Q2: 120k}}` → Output: “Sales grew from $100,000 in Q1 to $120,000 in Q2, a 20% increase.”
- **Advanced** : Input: `{patient: 'John', BP: '140/90'}` → Output: “John’s blood pressure is 140/90; consult a doctor.”
- **Edge Case** : Input: `{temp: -500°C}` → Output: “Invalid temperature detected.”

### 2.6 Thought Experiment

Imagine you’re a data point in a giant spreadsheet, like a star in the sky. NLG is the astronomer who describes your position in words, making you understandable without the raw numbers. Execution-based evaluation checks if the description leads to the right star when followed.

### 2.7 Scientific Connection

NLG automates tasks like writing experiment summaries, freeing you to focus on discoveries, like Franklin analyzing DNA patterns. Use it to describe physics data or generate code for simulations.

---

## Chapter 3: The Need for Evaluation – Ensuring Quality

### 3.1 What is Evaluation?

Evaluation checks if an NLG system produces high-quality text. We assess:

- **Fluency** : Does it read smoothly, like human writing?
- **Accuracy** : Are the facts correct?
- **Usefulness** : Does it solve a problem (e.g., answer a question)?
- **Coherence** : Does it make sense as a whole?
- **Feynman Analogy** : Evaluation is like tasting a cake to check if the recipe (NLG) worked, not just if it looks pretty.

### 3.2 Why Evaluate?

Without evaluation, you might trust an NLG system that produces wrong or useless outputs, like building a rocket without testing it. Evaluation ensures reliability, like Newton dropping apples to confirm gravity.

- **Einstein Thought Experiment** : If an NLG writes “The sky is green,” how do you know it’s wrong without checking the real sky? Evaluation is the check.

### 3.3 Types of Evaluation

- **Intrinsic** : Tests text quality (e.g., grammar, style). Example: BLEU score compares NLG text to human references.
- **Extrinsic** : Tests task performance (e.g., does a chatbot book a ticket?). Measured by success rate.
- **Execution-Based** : Runs NLG output as a program (e.g., SQL query) and checks results. Our focus!

### 3.4 History of Evaluation

- **1950s** : Turing’s “imitation game” asked if machines could fool humans.
- **1980s** : BLEU and ROUGE metrics quantified text similarity.
- **2000s** : Extrinsic metrics tested task performance (e.g., question answering).
- **2018** : Datasets like Spider introduced execution-based evaluation for text-to-SQL.
- **Why History?** : Like Darwin’s fossils, past methods show what works and what doesn’t.

### 3.5 Mathematics: Measuring Human Agreement

When humans judge NLG (e.g., “Is this fluent?”), we use **Cohen’s Kappa** to measure agreement:

- **Formula** :
  [
  \kappa = \frac{P_o - P_e}{1 - P_e}
  ]
- ( P_o ): Observed agreement (% of times judges agree).
- ( P_e ): Expected agreement by chance.
- **Example** :
- Two judges rate 10 outputs, agree on 8 (( P_o = 0.8 )).
- Chance agreement (random guessing) = 0.5 (( P_e = 0.5 )).
- ( \kappa = \frac{0.8 - 0.5}{1 - 0.5} = \frac{0.3}{0.5} = 0.6 ) (moderate agreement).
- **Derivation** :
- ( P_e = \sum (\text{probability of each category})^2 ).
- For two categories (good/bad), each 50% likely: ( P_e = 0.5^2 + 0.5^2 = 0.5 ).
- **Why?** : Like Curie ensuring lab assistants saw the same results, this ensures reliable judgments.

### 3.6 Examples

- **Good Output** : “The sun shines brightly.” (Fluent, accurate.)
- **Bad Output** : “Sun the brightly shines.” (Not fluent, confusing.)
- **Execution-Based** : SQL query `SELECT COUNT(*) FROM employees` returns 5, matches truth.

### 3.7 Scientific Connection

Evaluation is your scientific method for NLG. It’s like Einstein testing relativity with experiments—without it, your AI might mislead you in research.

---

## Chapter 4: Execution-Based Evaluation – The Core Concept

### 4.1 What is Execution-Based Evaluation?

Execution-based evaluation tests NLG by treating its output as executable instructions (e.g., SQL queries, Python code, API calls) and checking if the results are correct. It’s ideal for NLG generating functional outputs, like database queries or code.

- **Example** : Question: “How many employees in sales?” NLG generates `SELECT COUNT(*) FROM employees WHERE department='sales';`. Run it, check if it returns the correct number (e.g., 10).
- **Turing Analogy** : Like testing a computer’s program to see if it solves the problem, not just if it looks right.

### 4.2 Why It’s Unique

Unlike intrinsic (text quality) or extrinsic (task success) evaluations, execution-based evaluation tests _functionality_ . It ensures the output does what it’s supposed to, like baking a cake from an NLG recipe and checking if it’s chocolate.

- **Thought Experiment** : If NLG writes code to calculate 2+2, does it give 4? Run it to find out!

### 4.3 History and Evolution

- **1970s** : Denotational semantics linked words to actions (e.g., a sentence’s meaning is its effect).
- **1980s** : Semantic parsing turned questions into database queries (e.g., GeoQuery dataset).
- **2018** : Spider dataset standardized execution-based evaluation with 10,000 questions across 200 databases.
- **2020s** : Code generation (e.g., GitHub Copilot) adopted similar methods.

### 4.4 Deep Dive: Why It Matters

Execution-based evaluation is critical for applications where correctness is non-negotiable, like:

- **Business** : Querying sales data accurately.
- **Science** : Analyzing experiment results (e.g., particle physics).
- **Ethics** : Ensuring outputs don’t cause harm (e.g., wrong medical queries).

### 4.5 Examples for All Levels

- **Beginner** : Question: “Count employees.” Query: `SELECT COUNT(*) FROM employees;`. Result: 5, matches truth.
- **Intermediate** : Question: “Names in IT.” Query: `SELECT name FROM employees WHERE department='IT';`. Result: ['Charlie', 'Diana'].
- **Advanced** : Question: “Average salary by department.” Query: `SELECT department, AVG(salary) FROM employees GROUP BY department;`. Checks averages.
- **Edge Case** : Question: “Invalid department.” Query: `SELECT * FROM employees WHERE department='XYZ';`. Returns empty, handled correctly.

### 4.6 Scientific Connection

Like Franklin’s precise DNA measurements, execution-based evaluation ensures your NLG produces reliable results for experiments. Use it to query lab data or generate simulation code.

---

## Chapter 5: How It Works – Step-by-Step Process

### 5.1 The Process

Execution-based evaluation follows these steps:

1. **Generate Output** : NLG creates an executable output (e.g., SQL query).
2. **Check Syntax** : Ensure it’s valid (no errors).
3. **Execute** : Run it in an environment (e.g., database, interpreter).
4. **Compare Results** : Match output to expected truth.
5. **Score** : Calculate metrics like EX or F1.

- **Curie Analogy** : Like setting up a lab experiment, running it, and checking results against a hypothesis.

### 5.2 Detailed Sub-Steps

- **Syntax Check** : Use parsers (e.g., SQL validator) to catch errors like `SELCT` instead of `SELECT`.
- **Execution Environment** : Use a sandbox (e.g., SQLite in-memory database, as in `setup_database.py`) to run safely.
- **Comparison** : Check exact match (e.g., same number) or partial match (e.g., same rows, different order).
- **Scoring** : Apply metrics (see Chapter 6).
- **Error Handling** : Catch runtime errors (e.g., timeouts, invalid inputs).

### 5.3 Challenges

- **Syntax Errors** : Queries that don’t run (e.g., wrong keywords).
- **Ambiguity** : Questions with multiple interpretations (e.g., “top employees” by salary or tenure?).
- **Efficiency** : Slow queries on large databases.
- **Environment Setup** : Ensuring consistent databases (addressed in `setup_database.py`).

### 5.4 Example Workflow

- **Question** : “How many employees in HR?”
- **NLG Output** : `SELECT COUNT(*) FROM employees WHERE department='HR';` (simulated in `mini_project.py`).
- **Syntax Check** : Valid SQL.
- **Execution** : Run in SQLite (see `evaluate_nlg.py`).
- **Result** : Returns 2, matches true answer.
- **Score** : Counts as correct for EX calculation.

### 5.5 Scientific Connection

This process is like an experiment in your lab: design (generate), test (execute), and validate (compare). Use `mini_project.py` to practice this workflow.

---

## Chapter 6: Metrics and Mathematics – Measuring Success

### 6.1 Key Metrics

Execution-based evaluation uses metrics to quantify performance:

- **Execution Accuracy (EX)** :
- **What** : % of outputs that run and match true results.
- **Formula** : ( \text{EX} = \frac{\text{Correct}}{\text{Total}} \times 100% )
- **Example** : 8/12 queries correct → ( \text{EX} = \frac{8}{12} \times 100 = 66.67% ).
- **Calculation** : ( 8 \div 12 = 0.6667 ), ( 0.6667 \times 100 = 66.67 ).
- **Valid Efficiency Score (VES)** :
- **What** : % of outputs that run correctly and efficiently (e.g., under 1 second).
- **Example** : 10 queries, 6 valid, 4 efficient → ( \text{VES} = \frac{4}{10} \times 100 = 40% ).
- **Test-Suite Accuracy (TS)** :
- **What** : Average correctness across test cases.
- **Formula** : ( \text{TS} = \frac{\text{Matches}}{\text{Tests}} )
- **Example** : 5 tests, 4 correct → ( \text{TS} = \frac{4}{5} = 0.8 ) (80%).
- **F1 Score** :
- **What** : Balances precision (correct outputs given) and recall (correct outputs not missed).
- **Formula** : ( \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} )
- **Precision** : ( \frac{\text{True Positives (TP)}}{\text{TP} + \text{False Positives (FP)}} )
- **Recall** : ( \frac{\text{TP}}{\text{TP} + \text{False Negatives (FN)}} )
- **Example** : Generated 5 rows, 3 correct (TP=3), 2 wrong (FP=2), true has 4 rows (FN=1).
  - Precision: ( \frac{3}{3+2} = 0.6 )
  - Recall: ( \frac{3}{3+1} = 0.75 )
  - F1: ( 2 \times \frac{0.6 \times 0.75}{0.6 + 0.75} = \frac{0.9}{1.35} \approx 0.667 )

### 6.2 Advanced Metric: BLEU vs. Execution-Based

- **BLEU (Bilingual Evaluation Understudy)** : Intrinsic metric comparing NLG text to human references.
- **Formula** : Geometric mean of n-gram precisions (1- to 4-grams) × brevity penalty.
- **Example** : Generated: “Cat sat.” Reference: “The cat sat.” 1-gram precision = 2/2 = 1, brevity penalty = 0.67 (short), BLEU ≈ 0.67.
- **Limitation** : Doesn’t test functionality, unlike EX.
- **Why Execution-Based?** : Tests if output _works_ (e.g., correct query results), critical for scientific applications.

### 6.3 Derivations

- **F1 Derivation** :
- Precision measures how many generated outputs are correct.
- Recall measures how many true outputs were captured.
- F1 is the harmonic mean, balancing both for a fair score.
- Harmonic mean derivation: ( \text{F1} = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}} ).
- **Perplexity Derivation** :
- Measures model uncertainty: ( \text{PP} = 2^{\text{H}(W)} ), where ( \text{H}(W) = -\frac{1}{n} \sum \log_2 P(w_i) ).
- Lower perplexity means the model is more confident (better).

### 6.4 Examples

- **EX** : 15/20 queries correct → ( \text{EX} = 75% ).
- **F1** : TP=4, FP=1, FN=2 → Precision=0.8, Recall=0.667, F1≈0.727.
- **Edge Case** : Query fails to run (syntax error) → EX=0, F1=0.

### 6.5 Scientific Connection

Metrics are your scientific ruler, like a mathematician’s proof. Use `evaluate_nlg.py` to calculate these and validate your NLG system for research papers.

---

## Chapter 7: Practical Implementation – Code and Experiments

### 7.1 Setting Up a Test Environment

To test execution-based evaluation, you need a controlled environment, like a SQLite database. The `setup_database.py` file creates an in-memory database with an `employees` table:

- **Fields** : id, name, department, salary.
- **Sample Data** : Alice (Sales, $50k), Bob (Sales, $55k), Charlie (IT, $60k), Diana (IT, $65k).

### 7.2 Generating and Evaluating Queries

Use a simple NLG function to simulate query generation (extendable with `transformers` for real models). Evaluate with `evaluate_nlg.py`:

- **Steps** :

1. Generate SQL (e.g., `SELECT COUNT(*) FROM employees WHERE department='Sales';`).
2. Check syntax using SQLite parser.
3. Run in database.
4. Compare to true result (e.g., 2 employees).
5. Calculate EX, F1 (see `evaluate_nlg.py`).

### 7.3 Example Code

```python
# From evaluate_nlg.py
import sqlite3
from setup_database import create_test_database

def evaluate_query(conn, sql_query, true_result):
    cursor = conn.cursor()
    try:
        result = cursor.execute(sql_query).fetchall()
        result = result[0][0] if len(result) == 1 and isinstance(true_result, int) else [r[0] for r in result]
        return result == true_result
    except sqlite3.Error as e:
        print(f"Error: {e}")
        return False

conn = create_test_database()
sql = "SELECT COUNT(*) FROM employees WHERE department='Sales';"
true_result = 2
print("Correct:", evaluate_query(conn, sql, true_result))
conn.close()
```

### 7.4 Advanced Implementation

- **Real NLG Model** : Use T5 (from `transformers`) for text-to-SQL:

```python
  from transformers import T5Tokenizer, T5ForConditionalGeneration
  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  model = T5ForConditionalGeneration.from_pretrained('t5-small')
  input_text = "translate to SQL: How many employees in Sales?"
  inputs = tokenizer(input_text, return_tensors="pt")
  outputs = model.generate(**inputs)
  sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

- **Run** : Execute with `evaluate_nlg.py`, fine-tune on Spider dataset.

### 7.5 Scientific Connection

Coding is your lab experiment. Use `mini_project.py` to practice and `evaluate_nlg.py` to test real models, like Curie refining her radiation measurements.

---

## Chapter 8: Visualizing Results – Making Insights Clear

### 8.1 Why Visualize?

Visualizations make NLG performance intuitive, like Einstein’s thought experiments showing relativity. They help identify trends (e.g., which queries fail) and communicate results.

### 8.2 Types of Visualizations

- **Bar Plot** : Compare EX and F1 across queries (see `visualize_metrics.py`).
- **Flowchart** : Show evaluation process: Question → NLG → Query → Database → Result.
- **Heatmap** : Show error types (e.g., syntax vs. logic errors).

### 8.3 Example Visualization

```python
# From visualize_metrics.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.DataFrame({
    'Query': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] * 2,
    'Score': [70, 80, 60, 90, 85] + [65, 75, 55, 88, 82],
    'Metric': ['EX'] * 5 + ['F1'] * 5
})
sns.barplot(x='Query', y='Score', hue='Metric', data=data)
plt.ylabel('Score (%)')
plt.title('Execution-Based Evaluation Metrics')
plt.show()
```

### 8.4 Described Visual

- **Flowchart** : Imagine a pipeline:
- Box: “Question” (e.g., “How many in Sales?”) → Arrow to “NLG Model” → Box: “SQL Query” → Arrow to “Database” → Box: “Results” → Arrow to “Compare with Truth” → Checkmark (correct) or X (wrong).
- **To Draw** : Sketch arrows connecting boxes, or use `matplotlib` to render (extend `visualize_metrics.py`).

### 8.5 Scientific Connection

Visualizations are your scientific posters, communicating findings clearly. Use `visualize_metrics.py` to plot real results and impress at conferences.

---

## Chapter 9: Real-World Applications – Impact in Action

### 9.1 Applications

Execution-based evaluation ensures NLG works in critical domains:

- **Business** : Automated sales reports (e.g., Amazon querying revenue).
- **Healthcare** : Querying patient data (e.g., average blood pressure).
- **Science** : Analyzing experiment data (e.g., CERN particle speeds).
- **Accessibility** : Generating API calls for virtual assistants.
- **Education** : Producing code examples for students (e.g., GitHub Copilot).

### 9.2 Detailed Examples

- **Business** : Question: “Top 3 products by sales.” Query: `SELECT product, SUM(sales) FROM orders GROUP BY product ORDER BY SUM(sales) DESC LIMIT 3;`. Evaluated with EX (see `Case_Studies.md`).
- **Science** : Question: “Average particle speed in 2020 experiments.” Query: `SELECT AVG(speed) FROM experiments WHERE year=2020;`. Critical for accurate research.

### 9.3 Ethical Considerations

- **Accuracy** : Wrong queries can mislead (e.g., incorrect medical data).
- **Safety** : Prevent harmful executions (e.g., deleting databases).
- **Bias** : Ensure NLG doesn’t misinterpret sensitive questions.

### 9.4 Scientific Connection

Applications automate repetitive tasks, letting you focus on discoveries, like Lovelace envisioning computational creativity. See `Case_Studies.md` for full details.

---

## Chapter 10: Research Directions – Pushing the Boundaries

### 10.1 Current Gaps

- **Complex Queries** : NLG struggles with nested queries or multi-table joins (e.g., Spider dataset challenges).
- **Generalization** : Models fail on unseen database schemas.
- **Ambiguity** : Handling vague questions (e.g., “top employees”).
- **Efficiency** : Slow queries on large datasets.

### 10.2 Rare Insights

- **Execution vs. Understanding** : Perfect execution doesn’t mean the model understands, a philosophical question Turing posed.
- **Interdisciplinary Potential** : NLG could query biology or physics data, merging AI with science.
- **Ethical Risks** : Unchecked queries could cause harm (e.g., wrong medical advice).

### 10.3 Research Ideas

- **Hybrid Evaluation** : Combine execution-based with human ratings (e.g., fluency + EX).
- **Domain Adaptation** : Train NLG for specific fields (e.g., chemistry).
- **Quantum NLG** : Generate instructions for quantum computers.
- **Robustness** : Test NLG on noisy or incomplete data.

### 10.4 Scientific Connection

Like Einstein’s relativity, these ideas push NLG beyond current limits. Use `evaluate_nlg.py` to test new metrics or `mini_project.py` to explore domain-specific queries.

---

## Chapter 11: Projects – Hands-On Learning

### 11.1 Mini Project: Simple Text-to-SQL Evaluator

- **Objective** : Build a system to generate and evaluate SQL queries for an employee database.
- **Steps** :

1. Use `setup_database.py` to create a database with 5 employees.
2. Write a function to generate SQL for questions like “Count employees in HR” (see `mini_project.py`).
3. Evaluate 3 queries using EX and F1 (use `evaluate_nlg.py`).
4. Visualize results with `visualize_metrics.py`.

- **Example Question** : “List Sales employees.” Query: `SELECT name FROM employees WHERE department='Sales';`. True result: ['Alice', 'Bob'].

### 11.2 Major Project: Spider Dataset Analysis

- **Objective** : Test a transformer-based NLG model on a Spider dataset subset.
- **Steps** :

1. Download Spider (available online).
2. Fine-tune T5 (`transformers`) for text-to-SQL.
3. Run queries on SQLite databases (extend `setup_database.py`).
4. Evaluate with EX, VES, TS, F1 (use `evaluate_nlg.py`).
5. Plot results (use `visualize_metrics.py`).

- **Research Output** : Write a paper on improving EX for complex queries.

### 11.3 Scientific Connection

Projects are your lab experiments, like Curie’s radium tests. Start with the mini project to build skills, then tackle the major project for publication-worthy results.

---

## Chapter 12: Exercises – Build Your Skills

### 12.1 Beginner Exercises

1. **Calculate EX** : 9/15 queries correct. Answer: ( \text{EX} = \frac{9}{15} \times 100 = 60% ).
2. **Write SQL** : For “Count all employees.” Answer: `SELECT COUNT(*) FROM employees;`.
3. **Rate Fluency** : “Employees the in Sales is 2.” Answer: ~1/5 (poor grammar).

### 12.2 Intermediate Exercises

1. **Calculate F1** : TP=5, FP=2, FN=1. Answer: Precision=5/7≈0.714, Recall=5/6≈0.833, F1≈0.769.
2. **Generate Query** : For “Average salary in IT.” Answer: `SELECT AVG(salary) FROM employees WHERE department='IT';`.
3. **Run Code** : Use `evaluate_nlg.py` to test above query on a database.

### 12.3 Advanced Exercises

1. **Complex Query** : Write SQL for “Top 2 departments by employee count.” Answer: `SELECT department, COUNT(*) FROM employees GROUP BY department ORDER BY COUNT(*) DESC LIMIT 2;`.
2. **Error Handling** : Modify `evaluate_nlg.py` to log syntax errors.
3. **Visualize** : Plot EX for 10 queries using `visualize_metrics.py`.

### 12.4 Solutions

- **Exercise 1.1** : ( 9 \div 15 = 0.6 ), ( 0.6 \times 100 = 60 ).
- **Exercise 2.1** : Precision: ( \frac{5}{5+2} \approx 0.714 ), Recall: ( \frac{5}{5+1} \approx 0.833 ), F1: ( 2 \times \frac{0.714 \times 0.833}{0.714 + 0.833} \approx 0.769 ).

### 12.5 Scientific Connection

Exercises build your experimental skills, like Newton refining gravity laws through repeated tests. Use them to master evaluation techniques.

---

## Chapter 13: What’s Missing in Standard Tutorials – Filling the Gaps

### 13.1 Common Gaps

- **Shallow History** : Most tutorials skip NLG’s roots (e.g., semantic parsing, Turing’s ideas).
- **Limited Math** : Few derive metrics like F1 or perplexity fully.
- **Basic Examples** : Lack advanced or edge cases (e.g., invalid inputs).
- **No Ethics** : Ignore risks like harmful queries.
- **Narrow Applications** : Miss interdisciplinary uses (e.g., science, quantum computing).
- **Static Code** : Lack modular, reusable code like `setup_database.py`.

### 13.2 How This Book Fills Them

- **Deep History** : Covers NLG from Turing to transformers (Chapter 2).
- **Full Math** : Derives EX, F1, perplexity with step-by-step examples (Chapter 6).
- **Rich Examples** : Includes beginner to edge cases (Chapters 2, 4).
- **Ethics** : Discusses safety and bias (Chapter 9).
- **Interdisciplinary** : Explores science, healthcare, education (Chapter 9, `Case_Studies.md`).
- **Practical Code** : Provides modular `.py` files for experiments (Chapter 7).

### 13.3 Scientific Connection

Filling gaps is like Curie discovering new elements—addressing what others miss drives breakthroughs. Use this to design novel NLG experiments.

---

## Chapter 14: Future Directions – Your Path Forward

### 14.1 Next Steps

- **Learn** : Study Spider, WikiSQL datasets for text-to-SQL.
- **Experiment** : Fine-tune T5 with `transformers` (extend `mini_project.py`).
- **Visualize** : Use `visualize_metrics.py` to analyze trends.
- **Publish** : Write a paper on improving EX or handling ambiguity.
- **Explore** : Test NLG in new domains (e.g., biology, quantum computing).

### 14.2 Career Tips

- **Portfolio** : Build NLG projects using `.py` files to showcase at conferences.
- **Collaborate** : Join AI research communities (e.g., ACL, NeurIPS).
- **Ethics** : Ensure your NLG systems are safe and unbiased.

### 14.3 Philosophical Question

Does perfect execution mean the model understands? Turing would argue no—explore this in your research to push AI boundaries.

### 14.4 Scientific Connection

Like Einstein envisioning relativity’s future, these directions guide your NLG journey. Start with `mini_project.py`, then aim for a groundbreaking paper.

---

## Chapter 15: Conclusion – Becoming an NLG Scientist

Congratulations! You’ve journeyed through the world of execution-based evaluation in NLG, from fundamentals to cutting-edge research. You’ve learned:

- How NLG turns data into text, like a storyteller for numbers.
- Why execution-based evaluation ensures outputs work, like Turing’s functional tests.
- How to implement, evaluate, and visualize NLG with `.py` files.
- Real-world applications and research gaps to explore.

You’re now equipped to experiment like Curie, think like Einstein, and innovate like Lovelace. Use `Case_Studies.md` for inspiration, `Cheat_Sheet.md` for quick reference, and the `.py` files for hands-on practice. Your next steps:

- Run `mini_project.py` to build confidence.
- Tackle the Spider dataset for a major project.
- Publish a paper on improving NLG evaluation.

You’re on your way to becoming an NLG scientist! If you need more guidance, revisit this book or ask for tailored help. Keep exploring, and let’s push the boundaries of AI together!

---

## Appendices

### Appendix A: References to Artifacts

- **Jupyter Notebook** : `Execution_Based_Evaluation_NLG_Tutorial.ipynb` for interactive learning.
- **Python Files** :
- `setup_database.py`: Create test databases.
- `evaluate_nlg.py`: Calculate EX, F1 metrics.
- `visualize_metrics.py`: Plot results.
- `mini_project.py`: Run a simple evaluator.
- **Case Studies** : `Case_Studies.md` for real-world examples.
- **Cheat Sheet** : `Cheat_Sheet.md` for quick reference.

### Appendix B: Resources

- **Datasets** : Spider, WikiSQL (available online).
- **Libraries** : `sqlite3`, `pandas`, `matplotlib`, `seaborn`, `transformers`.
- **Papers** : “Attention is All You Need” (2017), Spider dataset paper (2018).

### Appendix C: Glossary

- **NLG** : Natural Language Generation, creating text from data.
- **Execution-Based Evaluation** : Testing NLG by running outputs and checking results.
- **EX** : Execution Accuracy, % of correct outputs.
- **F1** : Balances precision and recall for evaluation.
