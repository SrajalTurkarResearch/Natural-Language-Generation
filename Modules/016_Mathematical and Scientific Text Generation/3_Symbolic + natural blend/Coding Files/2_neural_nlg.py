# 2_neural_nlg.py
# Neural NLG: Using pre-trained T5 model for summarization
# Like a student who learned from thousands of examples

from transformers import pipeline


def neural_summary_nlg(text):
    """
    Generate summary using T5 transformer model.
    Input: long text
    Output: short, fluent summary
    """
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
    return summary[0]["summary_text"]


# === TEST ===
if __name__ == "__main__":
    text = "Apples are red, juicy fruits that grow on trees and are rich in fiber. They are healthy snacks."
    print("Input:", text)
    print("Summary:", neural_summary_nlg(text))
