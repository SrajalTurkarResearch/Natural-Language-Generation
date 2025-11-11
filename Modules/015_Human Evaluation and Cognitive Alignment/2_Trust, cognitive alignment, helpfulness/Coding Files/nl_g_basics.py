# nl_g_basics.py
# --------------------------------------------------------------
# NLG Fundamentals – text generation with HuggingFace transformers
# --------------------------------------------------------------

# 1. Install (uncomment the first time)
# --------------------------------------------------------------
# !pip install transformers torch matplotlib

# 2. Imports
# --------------------------------------------------------------
from transformers import pipeline
import matplotlib.pyplot as plt

# 3. Simple NLG pipeline (gpt2)
# --------------------------------------------------------------
generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

prompt = "In a world where machines can talk, trust is built by"
output = generator(prompt, max_length=80, num_return_sequences=1, temperature=0.7)[0][
    "generated_text"
]
print("\n=== Generated Text ===\n")
print(output)

# 4. Visualise word-probability intuition (beam-search demo)
# --------------------------------------------------------------
words = ["mat", "hat", "bat", "rat"]
probs = [0.60, 0.20, 0.10, 0.10]  # pretend these are the model logits

plt.figure(figsize=(6, 4))
plt.bar(words, probs, color="#4e79a7")
plt.title("Word-choice probabilities in NLG (illustrative)")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
