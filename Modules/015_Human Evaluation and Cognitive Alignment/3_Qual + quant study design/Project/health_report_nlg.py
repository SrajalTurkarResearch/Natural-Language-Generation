# health_report_nlg.py
# Real-World Use Case: Generate patient discharge summaries from structured EHR data
# Mixed Methods: Quant (BLEU) + Qual (theme extraction via keywords + word cloud)

from transformers import pipeline
import nltk
from nltk.translate.bleu_score import sentence_bleu
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json

nltk.download("punkt")

# Step 1: Simulated EHR Data (Realistic Structure)
ehr_data = {
    "patient_id": "P1001",
    "age": 68,
    "diagnosis": "Congestive Heart Failure",
    "medications": ["Lisinopril", "Furosemide", "Metoprolol"],
    "vitals": {"BP": "118/76", "HR": 82},
    "discharge_instructions": "Low sodium diet, daily weight check, follow-up in 7 days",
}

# Step 2: NLG Generation
generator = pipeline("text-generation", model="gpt2-medium")
prompt = f"""
Patient {ehr_data['patient_id']} is a {ehr_data['age']}-year-old with {ehr_data['diagnosis']}.
Current medications: {', '.join(ehr_data['medications'])}.
Vitals: BP {ehr_data['vitals']['BP']}, HR {ehr_data['vitals']['HR']}.
Discharge plan: {ehr_data['discharge_instructions']}.
Write a clear, empathetic discharge summary:
"""
output = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7)[0][
    "generated_text"
]
print("Generated Discharge Summary:")
print(output.split("Write a clear")[1] if "Write a clear" in output else output)
print("\n" + "=" * 60 + "\n")

# Step 3: Reference (Gold Standard)
reference = """
Patient P1001, a 68-year-old with congestive heart failure, is stable for discharge.
Continue Lisinopril, Furosemide, and Metoprolol. BP is 118/76, HR 82.
Instruct patient on low sodium diet and daily weight monitoring.
Follow-up appointment scheduled in 7 days.
"""

# Step 4: Quantitative Evaluation - BLEU
ref_tokens = [nltk.word_tokenize(reference.lower())]
cand_tokens = nltk.word_tokenize(output.lower())
bleu = sentence_bleu(ref_tokens, cand_tokens)
print(f"BLEU Score: {bleu:.4f}")

# Step 5: Qualitative Simulation - Extract Key Themes
qual_text = " ".join(
    [ehr_data["diagnosis"], ehr_data["discharge_instructions"]]
    + ehr_data["medications"]
)
wordcloud = WordCloud(width=600, height=400, background_color="white").generate(
    qual_text
)
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Qualitative Themes: Discharge Focus Areas")
plt.show()

# Research Insight
print("\nResearch Insight:")
print(
    "→ Quant: BLEU shows surface match. Qual: Word cloud reveals focus on 'heart', 'diet', 'follow-up'."
)
print("→ Next: Interview doctors — do they trust this summary?")
