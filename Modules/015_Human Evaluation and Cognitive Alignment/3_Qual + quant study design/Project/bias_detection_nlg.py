# bias_detection_nlg.py
# Use Case: Generate job ad → detect biased language
# Quant: Bias word count | Qual: Thematic analysis

from transformers import pipeline
import re

# Step 1: Job Data
job = {
    "role": "Software Engineer",
    "company": "TechCorp",
    "skills": ["leadership", "aggressive", "coding"],
}

# Step 2: Generate Job Ad
generator = pipeline("text-generation", model="gpt2")
prompt = f"Write a job ad for {job['role']} at {job['company']}. Required: {', '.join(job['skills'])}."
ad = generator(prompt, max_length=120)[0]["generated_text"]
print("Generated Job Ad:")
print(ad)
print("\n" + "=" * 60)

# Step 3: Bias Detection (Quant)
male_coded = ["strong", "aggressive", "leader", "dominant", "competitive"]
female_coded = ["collaborate", "empathy", "support", "nurture", "team player"]
m_count = sum(1 for w in male_coded if re.search(r"\b" + w + r"\b", ad, re.I))
f_count = sum(1 for w in female_coded if re.search(r"\b" + w + r"\b", ad, re.I))

print(f"Male-coded words: {m_count} | Female-coded: {f_count}")
bias_ratio = m_count / (m_count + f_count + 1)
print(f"Bias Score (Male-leaning): {bias_ratio:.2f}")

# Step 4: Qual Visualization
import matplotlib.pyplot as plt

plt.bar(["Male-Coded", "Female-Coded"], [m_count, f_count], color=["blue", "pink"])
plt.title("Gender Bias in NLG Job Ad")
plt.show()

print("\nEthics in Action: Retrain model with debiasing prompts.")
