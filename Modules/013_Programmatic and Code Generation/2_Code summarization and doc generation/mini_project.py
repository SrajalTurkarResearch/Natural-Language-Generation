# mini_project.py
# Purpose: Summarize a Fibonacci function to practice NLG skills.
# Why: Scientists summarize code (e.g., math models) to explain work clearly.
# Note: Manual summary here; use CodeT5 in real work for automation.


def fibonacci(n):
    """Computes nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Test the function
try:
    result = fibonacci(5)
    print(f"Fibonacci(5) = {result}")  # Output: 5
except Exception as e:
    print(f"Error: {e}")

# Manual summary (what AI would do)
summary = "Computes nth Fibonacci number recursively."
print("Summary:", summary)

# Why this matters: Summarizing math code helps explain models in papers.
# For science: Useful for algorithms in physics or biology.
# Try it: Write a new function (e.g., factorial) and summarize it manually.
