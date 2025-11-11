# symbolic_math_explainer.py
# Author: Grok (inspired by Turing, Einstein, Tesla)
# Description: Standalone script for symbolic math explanations using SymPy.
# Usage: python symbolic_math_explainer.py
# Dependencies: sympy (pip install sympy)

import sympy as sp


def symbolic_population_growth():
    """
    Symbolically represent and explain a population growth model.
    Formula: P(t) = P0 * e^(r t)
    """
    # Step 1: Define symbols
    t, P0, r = sp.symbols("t P0 r")  # t: time, P0: initial population, r: growth rate

    # Step 2: Create the expression
    P = P0 * sp.exp(r * t)  # exp is e^power

    # Step 3: Generate LaTeX and explanation
    latex_formula = sp.latex(P)
    explanation = (
        f"Formula (LaTeX): {latex_formula}\n"
        "Explanation: The population P at time t is the initial population P0 "
        "multiplied by e raised to the power of the growth rate r times time t. "
        "This models exponential growth, common in biology and finance."
    )

    return explanation


# Example usage
if __name__ == "__main__":
    print(symbolic_population_growth())
