# nlg_formula_explainer.py
# Author: Grok (inspired by Turing, Einstein, Tesla)
# Description: Standalone script to generate NLG explanations for math formulas using Transformers.
# Usage: python nlg_formula_explainer.py
# Dependencies: transformers (pip install transformers)

from transformers import pipeline


def explain_formula(formula):
    """
    Generate a natural language explanation for a given math formula.

    Args:
    formula (str): The mathematical formula (e.g., 'E = m * c^2').

    Returns:
    str: The generated explanation.
    """
    # Step 1: Load the text-to-text generation model (T5-base for explanations)
    explainer = pipeline("text2text-generation", model="t5-base")

    # Step 2: Create a prompt for the model
    prompt = f"Explain the mathematical formula: {formula}"

    # Step 3: Generate the explanation (limit length for concise output)
    try:
        explanation = explainer(prompt, max_length=100, num_return_sequences=1)[0][
            "generated_text"
        ]
    except Exception as e:
        explanation = f"Error generating explanation: {str(e)}"

    return explanation


# Example usage
if __name__ == "__main__":
    formula = "E = m * c^2"  # Change this to any formula
    print(f"Formula: {formula}")
    print("Generated Explanation:", explain_formula(formula))
    # Expected Output: Something like 'Energy equals mass times the speed of light squared, from relativity.'
