# finance_news_nlg_alignment.py
# --------------------------------------------------------------
# REAL-WORLD PROJECT: Generate earnings summaries + ALIGNMENT check
# Use case: Bloomberg Cyborg / Automated Insights
# --------------------------------------------------------------

# Install: pip install yfinance transformers scipy matplotlib numpy

import yfinance as yf
from transformers import pipeline
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. Fetch real stock data
# --------------------------------------------------------------
ticker = "AAPL"
data = yf.Ticker(ticker)
info = data.info
earnings = data.quarterly_earnings.iloc[-1]

revenue = earnings["Revenue"] / 1e9  # in billions
eps = earnings["Earnings"] / 1e6  # in millions

# --------------------------------------------------------------
# 2. NLG: Generate summary
# --------------------------------------------------------------
generator = pipeline("text-generation", model="gpt2")

prompt = f"Apple reported ${revenue:.2f}B revenue and ${eps:.1f}M EPS this quarter. Summarize for investors:"
summary = generator(prompt, max_length=100, temperature=0.7, num_return_sequences=1)[0][
    "generated_text"
]
summary = summary.split("Summarize for investors:")[-1].strip()
print("AI Summary:\n", summary)

# --------------------------------------------------------------
# 3. COGNITIVE ALIGNMENT: Compare word distribution
# --------------------------------------------------------------
human_summary = "Apple's revenue grew to $90.1 billion, beating estimates. EPS was strong at $1.5 billion."


def get_word_dist(text):
    words = text.lower().split()
    unique, counts = np.unique(words, return_counts=True)
    probs = counts / counts.sum()
    return dict(zip(unique, probs))


human_dist = get_word_dist(human_summary)
ai_dist = get_word_dist(summary)

# Align vocab
all_words = set(human_dist) | set(ai_dist)
p = np.array([human_dist.get(w, 1e-10) for w in all_words])
q = np.array([ai_dist.get(w, 1e-10) for w in all_words])

# Jensen-Shannon Divergence
m = 0.5 * (p + q)
jsd = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
alignment_score = 1 - np.sqrt(jsd)  # 1 = perfect alignment

print(f"\nCognitive Alignment Score: {alignment_score:.3f} (higher = better)")

# --------------------------------------------------------------
# 4. VISUALIZATION: Word distribution overlap
# --------------------------------------------------------------
common = set(human_dist) & set(ai_dist)
human_common = [human_dist[w] for w in common]
ai_common = [ai_dist[w] for w in common]

x = np.arange(len(common))
plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, human_common, 0.4, label="Human Analyst", color="#1f78b4")
plt.bar(x + 0.2, ai_common, 0.4, label="AI Summary", color="#33a02c")
plt.xticks(x, list(common), rotation=45)
plt.title("Cognitive Alignment: Word Usage Overlap")
plt.ylabel("Relative Frequency")
plt.legend()
plt.tight_layout()
plt.show()
