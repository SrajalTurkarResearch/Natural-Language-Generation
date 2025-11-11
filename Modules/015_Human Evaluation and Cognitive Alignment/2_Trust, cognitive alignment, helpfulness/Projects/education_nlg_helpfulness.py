# education_nlg_helpfulness.py
# --------------------------------------------------------------
# REAL-WORLD PROJECT: Personalized student feedback + HELPFULNESS score
# Use case: Duolingo, Khan Academy AI tutor
# --------------------------------------------------------------

# Install: pip install transformers nltk rouge-score matplotlib

from transformers import pipeline
import nltk
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

nltk.download("punkt")

# --------------------------------------------------------------
# 1. Student quiz data
# --------------------------------------------------------------
student = {"name": "Alex", "question": "What is 7 × 8?", "answer": "54", "correct": 56}

# --------------------------------------------------------------
# 2. NLG: Generate feedback
# --------------------------------------------------------------
generator = pipeline("text-generation", model="gpt2")

prompt = f"Student {student['name']} answered {student['answer']} to 7×8. Correct is {student['correct']}. Give helpful feedback:"
feedback = generator(prompt, max_length=80, temperature=0.7)[0]["generated_text"]
feedback = feedback.split("Give helpful feedback:")[-1].strip()
print("AI Feedback:\n", feedback)

# --------------------------------------------------------------
# 3. HELPFULNESS EVALUATION: ROUGE-L vs. ideal feedback
# --------------------------------------------------------------
ideal = "Close! 7×8 = 56. You wrote 54. Try: 7×7=49, then add 7 → 56. Great effort!"

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
scores = scorer.score(ideal, feedback)
helpfulness = scores["rougeL"].fmeasure
print(f"\nHelpfulness (ROUGE-L F1): {helpfulness:.3f}")

# --------------------------------------------------------------
# 4. VISUALIZATION: Helpfulness components
# --------------------------------------------------------------
components = ["Clarity", "Correctness", "Encouragement", "Actionable"]
values = [0.9, 0.7, 0.95, 0.8]  # manual rating or from LLM judge

plt.figure(figsize=(8, 5))
bars = plt.bar(components, values, color=["#1b9e77", "#d95f02", "#7570b3", "#e7298a"])
plt.title("Helpfulness Breakdown in AI Feedback")
plt.ylim(0, 1)
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.2f}",
        ha="center",
    )
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 5. Save
# --------------------------------------------------------------
with open("student_feedback.txt", "w") as f:
    f.write(
        f"Student: {student['name']}\nQuestion: {student['question']}\nAnswer: {student['answer']}\nFeedback: {feedback}\nHelpfulness: {helpfulness:.3f}"
    )
print("\nSaved to student_feedback.txt")
