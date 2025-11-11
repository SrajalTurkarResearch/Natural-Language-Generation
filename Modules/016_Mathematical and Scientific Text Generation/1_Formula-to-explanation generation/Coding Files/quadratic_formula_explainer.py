# quadratic_formula_explainer.py
# Author: Grok (inspired by Turing, Einstein, Tesla)
# Description: Standalone script for explaining the quadratic formula as an exercise.
# Usage: python quadratic_formula_explainer.py
# Dependencies: transformers (pip install transformers)

from transformers import pipeline


def explain_quadratic():
    """
    Generate an explanation for the quadratic formula using NLG.
    """
    # Step 1: Load model
    explainer = pipeline("text2text-generation", model="t5-base")

    # Step 2: Prompt for quadratic formula
    prompt = "Explain the mathematical formula: x = [-b ± sqrt(b^2 - 4ac)] / (2a)"

    # Step 3: Generate and return
    try:
        explanation = explainer(prompt, max_length=100, num_return_sequences=1)[0][
            "generated_text"
        ]
    except Exception as e:
        explanation = f"Error: {str(e)}"

    return explanation


# Run the exercise
if __name__ == "__main__":
    print("Quadratic Formula Explanation:", explain_quadratic())
    # Expected: Something like 'The solutions to ax^2 + bx + c = 0 are given by this formula, solving for x.'
