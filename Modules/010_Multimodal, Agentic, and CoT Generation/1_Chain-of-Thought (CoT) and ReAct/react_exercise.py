# react_exercise.py
# Purpose: Implement ReAct using SymPy for symbolic math.
# Context: Shows ReAct's ability to use external tools (SymPy) for reasoning in NLG.
# Problem: Solve x^2 - 5x + 6 = 0 symbolically.
# For scientists: This mirrors using lab tools to verify hypotheses.

from sympy import symbols, solve


def react_quadratic():
    """
    Uses ReAct to solve a quadratic equation with SymPy.
    Returns the reasoning process and answer.
    """
    steps = []
    # Thought: Define the problem
    steps.append("Thought: Solve x^2 - 5x + 6 = 0 symbolically.")
    # Action: Use SymPy as the tool
    x = symbols("x")
    eq = x**2 - 5 * x + 6
    result = solve(eq, x)
    # Observe
    steps.append(f"Observation: SymPy result = {result}")
    # Conclude
    steps.append("Answer: Roots are 2 and 3.")
    return "\n".join(steps)


if __name__ == "__main__":
    print("Running ReAct Exercise: Solve Quadratic Equation Symbolically")
    print(react_quadratic())

# Explanation for Aspiring Scientists:
# - ReAct uses tools (like SymPy) to ground reasoning, reducing errors in NLG.
# - This is like checking lab data to confirm a calculation.
# - Extend by using other tools (e.g., SciPy) or different equations.
