# projects_research.py
# A beginner-friendly tutorial on NLG evaluation metrics: Projects and Research Directions
# Includes mini/major project ideas and cutting-edge research insights

"""
Mini and Major Project Ideas
===========================
Mini Project: Chatbot Response Evaluator
--------------------------------------
Task: Evaluate chatbot responses using BLEU and BERTScore.
Steps:
1. Collect 10 chatbot responses and references.
2. Compute BLEU and BERTScore.
3. Visualize scores in a bar plot.
Code Snippet:
```python
from bert_score import score as bert_score
refs = ["How can I help you today?"]
cands = ["What can I do for you?"]
P, R, F1 = bert_score(cands, refs, lang="en")
print(f"BERTScore F1: {F1.mean().item():.3f}")
```

Major Project: NLG Model Benchmarking
------------------------------------
Task: Compare NLG models (e.g., GPT-2, T5) on summarization.
Steps:
1. Use CNN/DailyMail dataset.
2. Generate summaries with each model.
3. Compute BLEU, ROUGE, BERTScore, and human evaluations.
4. Analyze performance.
Code Snippet:
```python
from datasets import load_dataset
from transformers import pipeline
dataset = load_dataset('cnn_dailymail', '3.0.0', split='test[:10]')
summarizer = pipeline('summarization', model='t5-small')
summaries = [summarizer(article['article'], max_length=50)[0]['summary_text'] for article in dataset]
```

Research Directions
==================
1. Context-Aware Metrics: Develop metrics considering user intent or dialogue context.
2. Human-Metric Alignment: Improve automatic metrics to match human judgments.
3. Multimodal NLG: Evaluate text with images or audio.
4. Fairness and Bias: Detect bias in generated text.

Rare Insights
============
- BLEU penalizes valid paraphrases (e.g., "big" vs. "large"), less suitable for creative tasks.
- Human evaluators can be inconsistent due to biases (e.g., preferring formal language).
- Emerging metrics like BLEURT (Google, 2020) use fine-tuned models for better semantic understanding but are complex.

Instructions to Run
==================
1. Install libraries: `pip install datasets transformers bert-score`
2. Save as `projects_research.py`
3. Run: `python projects_research.py`
4. Outputs BERTScore for a sample chatbot response
"""

from bert_score import score as bert_score

# Sample mini project: Chatbot response evaluation
refs = ["How can I help you today?"]
cands = ["What can I do for you?"]
P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
print(f"BERTScore F1: {F1.mean().item():.3f}")

if __name__ == "__main__":
    print("NLG Evaluation Tutorial: Projects and Research Directions")
    print("Run complete. Check the BERTScore output.")
    print("Next: Run `future_tips.py` for future directions and tips.")
