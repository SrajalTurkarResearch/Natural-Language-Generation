# e2e_restaurant_nlg.py
# Real Dataset: E2E NLG Challenge[](https://github.com/tuetschek/e2e-cleaning)
# Task: Turn structured restaurant data into natural review

from datasets import load_dataset
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
import nltk
import matplotlib.pyplot as plt

nltk.download("punkt")

# Step 1: Load Real E2E Dataset
dataset = load_dataset("e2e_nlg", split="test[:5]")  # Small sample
sample = dataset[0]
print("Input (Meaning Representation):")
print(sample["meaning_representation"])
print("\nHuman Reference:")
print(sample["human_reference"])
print("\n" + "=" * 60)

# Step 2: NLG Generation
generator = pipeline("text-generation", model="gpt2")
prompt = f"Write a customer review based on: {sample['meaning_representation']}. Style: friendly, informative."
output = generator(prompt, max_length=80, temperature=0.8)[0]["generated_text"]
print("\nAI-Generated Review:")
print(output.split("Style:")[0] if "Style" in output else output)

# Step 3: BLEU Score
ref = [nltk.word_tokenize(sample["human_reference"].lower())]
cand = nltk.word_tokenize(output.lower())
bleu = sentence_bleu(ref, cand)
print(f"\nBLEU Score: {bleu:.4f}")

# Step 4: Qualitative Theme Match
themes = ["food", "price", "location", "service", "atmosphere"]
found = [t for t in themes if t in output.lower()]
print(f"Themes Covered: {found}")

# Plot
plt.bar(themes, [1 if t in output.lower() else 0 for t in themes], color="teal")
plt.title("Qualitative Theme Coverage in Generated Review")
plt.xticks(rotation=45)
plt.show()

print(
    "\nResearch Direction: Train fine-tuned model on E2E → improve BLEU + theme coverage."
)
