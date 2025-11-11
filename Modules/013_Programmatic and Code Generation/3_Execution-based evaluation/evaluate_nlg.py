# evaluate_nlg.py
"""
Purpose: Implement execution-based evaluation for NLG-generated SQL queries.
Evaluates queries by running them on a database and calculating metrics like
Execution Accuracy (EX) and F1 Score. Designed for aspiring scientists to test
NLG models in a research setting.

Dependencies: sqlite3 (built-in), setup_database.py (for database)

Usage: Run with a database connection from setup_database.py. Tests sample queries
and calculates metrics. Extend for research by adding more metrics or complex queries.

Author: Inspired by Turing’s computational precision and Feynman’s clear explanations.
"""

import sqlite3
from setup_database import create_test_database


def execution_accuracy(correct, total):
    """
    Calculate Execution Accuracy (EX): % of queries that run and match true results.
    Args:
        correct (int): Number of correct query results.
        total (int): Total queries tested.
    Returns:
        float: EX score as percentage.
    """
    return (correct / total) * 100


def f1_score(tp, fp, fn):
    """
    Calculate F1 Score: Balances precision and recall for query results.
    Args:
        tp (int): True positives (correct results returned).
        fp (int): False positives (incorrect results returned).
        fn (int): False negatives (correct results missed).
    Returns:
        float: F1 score.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )


def evaluate_query(conn, sql_query, true_result):
    """
    Run a single SQL query and check if it matches the true result.
    Args:
        conn: SQLite database connection.
        sql_query (str): SQL query to execute.
        true_result: Expected result (int or list).
    Returns:
        bool: True if result matches, False otherwise.
    """
    cursor = conn.cursor()
    try:
        result = cursor.execute(sql_query).fetchall()
        # Convert result to comparable format
        result = (
            result[0][0]
            if len(result) == 1 and isinstance(true_result, int)
            else [r[0] for r in result]
        )
        return result == true_result
    except sqlite3.Error as e:
        print(f"Error executing query '{sql_query}': {e}")
        return False


def main():
    # Setup database
    conn = create_test_database()

    # Sample questions and queries (simulating NLG output)
    questions = [
        "How many employees in Sales?",
        "List names in IT",
        "Average salary in IT",
    ]
    sql_queries = [
        "SELECT COUNT(*) FROM employees WHERE department='Sales';",
        "SELECT name FROM employees WHERE department='IT';",
        "SELECT AVG(salary) FROM employees WHERE department='IT';",
    ]
    true_results = [2, ["Charlie", "Diana"], 62500]

    # Evaluate queries
    correct = 0
    for i, (question, sql, true) in enumerate(
        zip(questions, sql_queries, true_results)
    ):
        print(f"\nTesting: {question}")
        is_correct = evaluate_query(conn, sql, true)
        print(f"Query: {sql}")
        print(f"Correct: {is_correct}")
        if is_correct:
            correct += 1

    # Calculate EX
    ex = execution_accuracy(correct, len(questions))
    print(f"\nExecution Accuracy: {ex:.2f}%")

    # Example F1 for one query (e.g., names in IT)
    tp = 2  # Both names correct
    fp = 0  # No wrong names
    fn = 0  # No missed names
    f1 = f1_score(tp, fp, fn)
    print(f"F1 Score for 'List names in IT': {f1:.3f}")

    conn.close()


if __name__ == "__main__":
    main()

# Scientist Tip: Add more queries or metrics (e.g., VES for efficiency) to test NLG
# robustness. Research idea: Evaluate queries on unseen databases to test generalization.
