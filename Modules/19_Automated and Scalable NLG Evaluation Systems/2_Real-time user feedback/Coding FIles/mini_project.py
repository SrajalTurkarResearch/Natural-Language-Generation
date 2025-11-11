# Mini Project: Simple Feedback Chatbot
# This script creates an interactive loop for text generation with user feedback.

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# Function to generate text with optional feedback
def generate_with_feedback(prompt, feedback=""):
    if feedback:
        prompt += f" {feedback}"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50)
    return tokenizer.decode(outputs[0])


# Interactive loop
prompt = input("Enter initial prompt: ")
output = generate_with_feedback(prompt)
print("Initial Output:", output)

feedback = input("Provide feedback to refine: ")
new_output = generate_with_feedback(prompt, feedback)
print("Refined Output:", new_output)
