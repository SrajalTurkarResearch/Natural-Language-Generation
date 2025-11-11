# legal_nlg_full_pipeline.py
# --------------------------------------------------------------
# REAL-WORLD PROJECT: Legal clause NLG with Trust + Alignment + Helpfulness
# Use case: LegalOn, ContractPodAi
# --------------------------------------------------------------

# Install: pip install transformers scikit-learn matplotlib

from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------
# 1. Input: Contract requirement
# --------------------------------------------------------------
requirement = "Payment must be made within 30 days of invoice."

# --------------------------------------------------------------
# 2. NLG: Generate legal clause
# --------------------------------------------------------------
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

prompt = f"Draft a clear legal clause: {requirement}\nClause:"
clause = generator(prompt, max_length=100, temperature=0.5)[0]["generated_text"]
clause = clause.split("Clause:")[-1].strip()
print("Generated Clause:\n", clause)

# --------------------------------------------------------------
# 3. TRUST: Fact-check key terms
# --------------------------------------------------------------
required_terms = ["payment", "30 days", "invoice"]
trust_acc = sum(1 for term in required_terms if term in clause.lower()) / len(
    required_terms
)
transparency = 0.85  # Model shows prompt
reliability = 0.9
trust_score = (trust_acc + transparency + reliability) / 3
print(f"Trust Score: {trust_score:.3f}")

# --------------------------------------------------------------
# 4. ALIGNMENT: Compare style to human lawyer
# --------------------------------------------------------------
human_clause = "The Buyer shall make payment in full within thirty (30) days of receipt of the invoice."
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([clause, human_clause])
cosine_sim = np.dot(vectors[0], vectors[1].T).toarray()[0][0]
alignment = cosine_sim
print(f"Cognitive Alignment (Cosine): {alignment:.3f}")

# --------------------------------------------------------------
# 5. HELPFULNESS: Clarity & Readability
# --------------------------------------------------------------
words = len(clause.split())
sentences = len(clause.split("."))
flesch = (
    206.835
    - 1.015 * (words / sentences)
    - 84.6 * (sum(len(w) for w in clause.split()) / words)
)
helpfulness = min(flesch / 100, 1.0)  # simplified
print(f"Helpfulness (Readability): {helpfulness:.3f}")

# --------------------------------------------------------------
# 6. FINAL DASHBOARD
# --------------------------------------------------------------
metrics = ["Trust", "Alignment", "Helpfulness"]
scores = [trust_score, alignment, helpfulness]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, scores, color=["#e41a1c", "#377eb8", "#4daf4a"])
plt.title("Legal NLG: Full Pillar Evaluation", fontsize=14)
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{scores[i]:.2f}",
        ha="center",
        fontweight="bold",
    )
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 7. Save
# --------------------------------------------------------------
with open("legal_clause.txt", "w") as f:
    f.write(
        f"Requirement: {requirement}\nAI Clause: {clause}\nTrust: {trust_score:.2f} | Alignment: {alignment:.2f} | Helpfulness: {helpfulness:.2f}"
    )
print("\nLegal NLG pipeline complete.")
