# doc_generation.py
# Purpose: Demonstrates creating a detailed docstring for a scientific function.
# Why: Scientists need clear notes for code (e.g., physics models) to share and repeat experiments.
# Note: This is manual; AI tools like CodeLlama would automate this.


def simulate_gravity(mass, height):
    """
    Calculates potential energy under gravity.

    Args:
        mass (float): Mass in kilograms.
        height (float): Height in meters.

    Returns:
        float: Energy in Joules.

    Example:
        >>> simulate_gravity(1, 10)
        490.5
    """
    g = 9.81  # Gravity constant (m/s^2)
    return 0.5 * mass * g * height**2


# Test the function
try:
    result = simulate_gravity(1, 10)
    print(f"Potential Energy for mass=1kg, height=10m: {result} Joules")
except Exception as e:
    print(f"Error running function: {e}")

# Why this matters: Clear docs are like lab reports, helping others understand your work.
# For science: Use for physics or engineering code to explain models clearly.
# Try it: Add an error check (e.g., raise ValueError if mass < 0) and update docstring.
