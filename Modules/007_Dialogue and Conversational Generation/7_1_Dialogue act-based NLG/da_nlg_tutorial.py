# da_nlg_tutorial.py
# Dialogue Act-Based NLG: Tutorial Module
# Author: Grok, inspired by Alan Turing, Albert Einstein, and Nikola Tesla
# Date: August 22, 2025
# Purpose: A modular Python file providing a step-by-step tutorial for DA-NLG.

"""
Tutorial: Dialogue Act-Based NLG Process

Step-by-Step:
1. Identify Dialogue Act (DA):
   - Use Natural Language Understanding (NLU) to classify input (e.g., 'What's the time?' → Request).
2. Content Planning:
   - Select relevant data (e.g., Time: 3 PM).
3. Sentence Planning:
   - Choose tone and structure (e.g., formal: 'The time is 3 PM.').
4. Surface Realization:
   - Generate text using templates or neural models.

Analogy (Tesla-Inspired):
Like optimizing alternating current, DA-NLG ensures efficient conversational flow, minimizing latency and maximizing clarity.

Example:
- Input: 'What's the weather?'
- DA: Request
- Content: {Weather: Sunny, Temp: 25°C}
- Output: 'The weather is sunny with a temperature of 25°C.'
"""


def run_tutorial_example(input_text):
    """Demonstrate a simple DA-NLG process."""
    if "weather" in input_text.lower():
        da = "Request"
        content = {"Weather": "Sunny", "Temp": "25°C"}
        output = f"The weather is {content['Weather']} with a temperature of {content['Temp']}."
    else:
        output = "Sorry, I don't understand."
    return f"Input: {input_text}\nDA: {da}\nOutput: {output}"


if __name__ == "__main__":
    print(run_tutorial_example("What's the weather?"))
