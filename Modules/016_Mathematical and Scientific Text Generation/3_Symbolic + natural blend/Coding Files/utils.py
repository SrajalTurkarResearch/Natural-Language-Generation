# utils.py
# Shared helper functions

from transformers import pipeline

# Global summarizer to avoid reloading
_summarizer = None


def neural_summary_nlg(text):
    """Cached neural summarization."""
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="t5-small")
    summary = _summarizer(text, max_length=50, min_length=10, do_sample=False)
    return summary[0]["summary_text"]
