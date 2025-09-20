# cot_example.py
# Purpose: Simulate Chain-of-Thought (CoT) reasoning for a simple math problem.
# Context: Demonstrates how CoT breaks down a problem into clear steps, as used in NLG to improve reasoning transparency.
# Problem: Roger has 5 tennis balls, buys 2 cans of 3 balls each. How many total?
# For scientists: This mirrors logical decomposition in research, like breaking down a hypothesis into testable steps.


def cot_example(problem):
    """
    Simulates CoT by generating step-by-step reasoning for a given problem.
    Returns a string with steps and final answer.
    """
    steps = []
    # Step 1: Understand the problem
    steps.append("Step 1: Understand - Roger has 5 balls, buys 2 cans of 3 each.")
    # Step 2: Calculate new balls
    new_balls = 2 * 3
    steps.append(f"Step 2: Compute new balls = {new_balls}")
    # Step 3: Compute total
    total = 5 + new_balls
    steps.append(f"Step 3: Total = {total}")
    return "\n".join(steps) + f"\nAnswer: {total}"


if __name__ == "__main__":
    # Example usage
    print("Running CoT Example for Roger Tennis Balls Problem")
    print(cot_example("Roger tennis balls"))

# Explanation for Aspiring Scientists:
# - CoT is like writing a lab notebook: each step is explicit, making reasoning reproducible.
# - In NLG, this ensures generated text (e.g., explanations) is logical and verifiable.
# - Try modifying this for other problems, e.g., change numbers or add steps.
