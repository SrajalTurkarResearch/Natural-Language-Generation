# NLG with Real-Time Feedback Simulation
# This script shows how to incorporate user feedback by modifying prompts.

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# Function to generate text, optionally with feedback
def generate_with_feedback(prompt, feedback=""):
    if feedback:
        prompt += f" {feedback}"  # Append feedback to refine the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")  # Encode prompt
    outputs = model.generate(inputs, max_length=50)  # Generate text
    return tokenizer.decode(outputs[0])  # Decode and return


# Example usage
initial_prompt = "Write a story about a robot."
print("Initial Generation:", generate_with_feedback(initial_prompt))
print("With Feedback:", generate_with_feedback(initial_prompt, "Make it adventurous."))
