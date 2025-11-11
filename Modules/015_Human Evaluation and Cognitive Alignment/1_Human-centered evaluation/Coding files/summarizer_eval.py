# summarizer_eval.py: NLG Summarization with Basic Evaluation
# Theory: Summarization NLG compresses text; evaluate intrinsically (fluency) or extrinsically (usefulness).
# Logic: Use pre-trained model; manual rating simulates human-centered step.
# Future Direction: Integrate with crowdsourcing (e.g., MTurk) for scaled human evals.

from transformers import pipeline

# Load summarization pipeline (BART or T5-based)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example long text
text = """
Natural Language Generation (NLG) is a subfield of AI where machines produce human-like text.
Evaluation traditionally uses metrics like BLEU, but human-centered approaches involve user studies.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)[0][
    "summary_text"
]
print("Generated Summary:", summary)

# Simulate human-centered eval: Manual Likert scale (1-5 for informativeness)
# In practice: Collect from users; here, placeholder
human_rating = 4  # Example: Rate based on criteria
print(f"Simulated Human Rating (1-5): {human_rating}")

# Major Project Tip: Load real dataset via 'datasets' library: from datasets import load_dataset; ds = load_dataset('cnn_dailymail')
