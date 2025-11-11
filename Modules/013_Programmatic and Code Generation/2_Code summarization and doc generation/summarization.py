# summarization.py
# Purpose: Shows how to summarize code using NLG, turning code into a short sentence.
# Why: Scientists need quick explanations of code (e.g., for experiment scripts) to save time.
# Note: Uses a general model (BART) for demo; in real work, use CodeT5 for code tasks.

from transformers import pipeline

# Initialize the summarization model
# Note: BART is used here for simplicity; CodeT5 is better for code but needs more setup.
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Example code to summarize
code = "def add(a, b): return a + b"

# Generate summary
# max_length=50: Keep it short; min_length=10: Ensure some detail; do_sample=False: Consistent output
try:
    summary = summarizer(code, max_length=50, min_length=10, do_sample=False)
    print("Code:", code)
    print("Summary:", summary[0]["summary_text"])
    # Expected: Something like "Adds two numbers and returns their sum."
except Exception as e:
    print(f"Error summarizing: {e}")

# Why this matters: In research, summarizing code (e.g., for DNA analysis) helps share findings fast.
# For science: Imagine summarizing a complex physics simulation to explain it in a paper.
# Try it: Change 'code' to another simple function (e.g., def square(n): return n*n) and run.
