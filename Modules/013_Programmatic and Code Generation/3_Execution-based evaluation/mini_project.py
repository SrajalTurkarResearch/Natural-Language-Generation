# mini_project.py
"""
Purpose: Implement a simple text-to-SQL evaluator for execution-based NLG evaluation.
Tests a small set of questions on an employee database, evaluating with EX.
Designed for aspiring scientists to practice NLG evaluation hands-on.

Dependencies: sqlite3, setup_database.py

Usage: Run to test 3 sample questions, generate SQL queries, and evaluate results.
Extend for research by adding complex queries or new metrics.

Author: Inspired by Curie’s experimental approach and Turing’s computational logic.
"""

import sqlite3
from setup_database import create_test_database
from evaluate_nlg import execution_accuracy, evaluate_query


def generate_sql(question):
    """
    Simulate NLG by generating SQL for simple questions.
    Args:
        question (str): Natural language question.
    Returns:
        str: SQL query or None if unsupported.
    """
    question = question.lower()
    if "count employees in hr" in question:
        return "SELECT COUNT(*) FROM employees WHERE department='HR';"
    elif "list sales employees" in question:
        return "SELECT name FROM employees WHERE department='Sales';"
    elif "count all employees" in question:
        return "SELECT COUNT(*) FROM employees;"
    return None


def main():
    # Setup database with different data for variety
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT)"
    )
    cursor.executemany(
        "INSERT INTO employees (name, department) VALUES (?, ?)",
        [("Eve", "HR"), ("Frank", "HR"), ("Grace", "Sales")],
    )
    conn.commit()

    # Test questions and expected results
    questions = ["Count employees in HR", "List Sales employees", "Count all employees"]
    true_results = [2, ["Grace"], 3]

    # Evaluate
    correct = 0
    for i, (question, true) in enumerate(zip(questions, true_results)):
        sql = generate_sql(question)
        if sql:
            print(f"\nTesting: {question}")
            is_correct = evaluate_query(conn, sql, true)
            print(f"Query: {sql}")
            print(f"Correct: {is_correct}")
            if is_correct:
                correct += 1
        else:
            print(f"\nTesting: {question}")
            print("Error: Could not generate SQL")

    # Calculate EX
    ex = execution_accuracy(correct, len(questions))
    print(f"\nMini Project Execution Accuracy: {ex:.2f}%")

    conn.close()


if __name__ == "__main__":
    main()

# Scientist Tip: Extend by integrating a real NLG model (e.g., T5 from transformers).
# Research idea: Test on a larger dataset or add error handling for robustness.
