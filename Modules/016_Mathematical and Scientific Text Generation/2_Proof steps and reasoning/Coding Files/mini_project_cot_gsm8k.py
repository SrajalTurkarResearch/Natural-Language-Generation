# mini_project_cot_gsm8k.py
# Mini Project: CoT on GSM8K Dataset (math reasoning).
# A small research experiment in logical NLG.

from transformers import pipeline  # For LLM

# Load model (reuse from cot_reasoning.py or standalone)
generator = pipeline("text-generation", model="gpt2")

# Sample from GSM8K (in real project, load full dataset via Hugging Face)
question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

# CoT Prompt (induce step-by-step math reasoning)
prompt = f"{question} Let's think step by step."
result = generator(prompt, max_length=100)
print(result[0]["generated_text"])
# Explanation: Applies CoT to math problems. For major research, evaluate accuracy on full dataset.
