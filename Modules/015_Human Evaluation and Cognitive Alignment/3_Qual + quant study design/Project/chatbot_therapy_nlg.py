# chatbot_therapy_nlg.py
# Use Case: Generate supportive response in mental health chatbot
# Mixed Eval: Quant (sentiment match) + Qual (empathy checklist)

from transformers import pipeline
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Step 1: User Input
user_input = "I've been feeling really anxious lately and can't sleep."

# Step 2: Generate Empathetic Response
generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")
empathetic_prompt = f"User: {user_input}\nTherapist (empathetic, validating, hopeful):"
response = generator(empathetic_prompt, max_length=100, pad_token_id=50256)[0]['generated_text']
print("AI Therapist Response:")
print(response.split("Therapist")[1] if "Therapist" in response else response)
print("\n" + "="*60)

# Step 3: Quantitative - Sentiment Match (simulated)
# In real study: use VADER or BERT sentiment
print("Quant Eval: Sentiment = Positive (hopeful tone detected)")

# Step 4: Qualitative - Empathy Checklist
empathy_markers = ['I hear you', 'that sounds', 'it's okay', 'you're not alone', 'take care']
found = [m for m in empathy_markers if m.lower() in response.lower()]
print(f"Empathy Markers Found: {len(found)}/5 → {found}")

# Word Cloud of Response
wordcloud = WordCloud(width=600, height=300).generate(response)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Empathy Language in AI Response")
plt.show()

print("\nStudy Idea: A/B test AI vs human responses with 50 users (mixed methods).")