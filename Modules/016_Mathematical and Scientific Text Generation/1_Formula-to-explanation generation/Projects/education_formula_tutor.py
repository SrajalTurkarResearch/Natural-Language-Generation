# education_formula_tutor.py
"""
Real-World Project: Interactive Formula Tutor
Use Case: Education – Explain any math formula to students in simple language.
Features: User inputs LaTeX or plain formula → Gets explanation + example.
"""

from transformers import pipeline
import sympy as sp

# Load NLG model
explainer = pipeline("text2text-generation", model="t5-base")


def explain_formula(user_formula):
    prompt = f"Explain this math formula to a high school student: {user_formula}"
    result = explainer(prompt, max_length=120, num_return_sequences=1)
    return result[0]["generated_text"]


def generate_example(formula_str):
    try:
        # Simple parsing for basic formulas
        if "=" in formula_str:
            lhs, rhs = formula_str.split("=", 1)
            expr = sp.sympify(rhs.strip())
            vars_dict = {str(v): 2 for v in expr.free_symbols}  # dummy values
            value = expr.subs(vars_dict)
            return f"Example: If {', '.join([f'{k}=2' for k in vars_dict])}, then {lhs.strip()} = {value}"
    except:
        return "No example generated."
    return "No example generated."


# Interactive Tutor
if __name__ == "__main__":
    print("Interactive Math Tutor (Type 'quit' to exit)\n")
    while True:
        formula = input("Enter a formula (e.g., E = m c^2): ").strip()
        if formula.lower() == "quit":
            break
        if not formula:
            continue
        explanation = explain_formula(formula)
        example = generate_example(formula)
        print("\nExplanation:")
        print(explanation)
        print(f"\n{example}\n")
        print("-" * 60)
