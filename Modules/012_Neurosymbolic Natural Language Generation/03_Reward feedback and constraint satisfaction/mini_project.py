# mini_project.py
# Purpose: Mini project to generate text with a length constraint
# Inspired by Tesla’s prototyping and Sutton’s iterative testing
# Use: Run to test a simple constrained text generator


def generate_text(length):
    # Dummy generator: Creates 'A' repeated by length
    # In practice, replace with a neural model (e.g., GPT-2)
    return "A" * length


# Define constraint: Length must be < 50
max_length = 50
test_length = 30
text = generate_text(test_length)

# Check constraint
if len(text) < max_length:
    print(f"Valid text (length {len(text)}): {text}")
else:
    print(f"Invalid text (length {len(text)} exceeds {max_length})")

# Explanation for researchers:
# - Simulates NLG with a constraint check
# - Real systems would use neural models with symbolic validators
# - Try changing max_length or generate_text to include words
# - Next step: Add reward feedback (e.g., score for keyword inclusion)
