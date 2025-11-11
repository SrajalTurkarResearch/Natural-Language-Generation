# Basic NLG Example using GPT-2
# This script demonstrates simple text generation without feedback.

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the prompt
prompt = "The future of AI is"

# Encode the prompt into tensor format for the model
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text using the model
outputs = model.generate(inputs, max_length=50)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0])
print(generated_text)
