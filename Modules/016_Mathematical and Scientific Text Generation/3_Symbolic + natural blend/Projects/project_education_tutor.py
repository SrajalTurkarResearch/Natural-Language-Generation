# project_education_tutor.py
# Real-World: AI Math Tutor with Step-by-Step Explanations
# Use: EdTech, Online Learning Platforms

import sympy as sp
from utils import neural_summary_nlg


def math_tutor_nlg(equation_str):
    """
    Solve equation symbolically, explain in natural language.
    """
    x = sp.symbols("x")
    eq = sp.sympify(equation_str)

    # === SYMBOLIC: Solve with SymPy ===
    try:
        solution = sp.solve(eq, x)
        steps = sp.latex(eq) + r" \rightarrow " + sp.latex(solution)
    except:
        return "Sorry, I couldn't solve this equation."

    # === NEURAL: Explain in words ===
    explanation = neural_summary_nlg(
        f"Solving the equation {equation_str}. "
        f"The solution is {solution}. "
        f"Steps involve algebraic manipulation."
    )

    return f"EQUATION: {equation_str}\nSOLUTION: x = {solution}\n\nEXPLANATION:\n{explanation}"


# === TEST ===
if __name__ == "__main__":
    print(math_tutor_nlg("x**2 - 5*x + 6 = 0"))
    print("\n" + "=" * 50 + "\n")
    print(math_tutor_nlg("x**3 - 8 = 0"))
