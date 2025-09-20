# cot_exercise.py
# Purpose: Implement CoT for solving a quadratic equation.
# Context: Demonstrates CoT for mathematical reasoning, critical for scientific NLG.
# Problem: Solve x^2 - 5x + 6 = 0.
# For scientists: This is like breaking down a complex experiment into logical steps.

import math


def solve_quadratic(a, b, c):
    """
    Uses CoT to solve a quadratic equation ax^2 + bx + c = 0.
    Returns steps as a string.
    """
    steps = []
    # Step 1: Calculate discriminant
    disc = b**2 - 4 * a * c
    steps.append(f"Step 1: Discriminant = {b}^2 - 4*{a}*{c} = {disc}")
    # Step 2: Check for real roots
    if disc < 0:
        return "No real roots"
    # Step 3: Compute roots
    root1 = (-b + math.sqrt(disc)) / (2 * a)
    root2 = (-b - math.sqrt(disc)) / (2 * a)
    steps.append(f"Step 2: Roots = {root1}, {root2}")
    return "\n".join(steps)


if __name__ == "__main__":
    print("Running CoT Exercise: Solve Quadratic Equation x^2 - 5x + 6 = 0")
    print(solve_quadratic(1, -5, 6))

# Explanation for Aspiring Scientists:
# - CoT here breaks down a math problem into clear, verifiable steps.
# - In NLG, this ensures explanations are structured and accurate.
# - Try changing coefficients (a, b, c) to test other equations.
