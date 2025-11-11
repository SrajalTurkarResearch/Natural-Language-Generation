# news_summary_nlg.py
# Use Case: Generate neutral news summary from structured event data
# Dataset-inspired: Like CNN/Daily Mail but simplified

from transformers import pipeline
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Step 1: Structured News Event
event = {
    "headline": "Tesla Reports Record Q3 Deliveries",
    "company": "Tesla",
    "deliveries": 462000,
    "revenue": "$24.3 billion",
    "location": "Fremont, CA",
    "ceo_quote": "We continue to improve efficiency.",
}

# Step 2: NLG Generation
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
generator = pipeline("text-generation", model="gpt2")
prompt = f"Write a neutral 2-sentence news summary: {event}"
gen = generator(prompt, max_length=100)[0]["generated_text"]
print("Generated News Summary:")
print(gen)
print("\n" + "=" * 60 + "\n")

# Step 3: Reference
ref = "Tesla delivered 462,000 vehicles in Q3, generating $24.3 billion in revenue. The company continues to scale production in Fremont."

# Step 4: ROUGE Evaluation
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = scorer.score(ref, gen)
print("ROUGE Scores:")
for k, v in scores.items():
    print(
        f"  {k}: Precision={v.precision:.3f}, Recall={v.recall:.3f}, F1={v.fmeasure:.3f}"
    )

# Step 5: Bias Check (Qualitative)
bias_keywords = ["amazing", "disaster", "revolutionary", "failed"]
found = [word for word in bias_keywords if word in gen.lower()]
print(f"\nBias Check: Found biased words: {found if found else 'None'}")

# Visualization
labels = ["ROUGE-1 F1", "ROUGE-L F1"]
values = [scores["rouge1"].fmeasure, scores["rougeL"].fmeasure]
plt.bar(labels, values, color=["skyblue", "lightcoral"])
plt.ylim(0, 1)
plt.title("News Summary Quality (Quant)")
plt.show()

print("\nNext Step: Run focus group — is the tone neutral enough?")
