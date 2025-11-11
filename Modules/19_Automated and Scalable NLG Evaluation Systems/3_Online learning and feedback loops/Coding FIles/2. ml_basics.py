"""
ml_basics.py
============
Module 2: How Machines Learn (with Math!)
Step-by-step gradient descent included.
"""

import numpy as np


def gradient_descent_example():
    """Full Math: Guessing Weight with Gradient Descent"""
    print("🧮 GRADIENT DESCENT EXAMPLE: Guessing Weight\n")

    w = 50.0  # Initial guess
    y_true = 70.0  # Real weight
    eta = 0.1  # Learning rate
    steps = 5

    print(f"True weight: {y_true} kg")
    print(f"Start guess: {w} kg\n")

    for step in range(steps):
        error = y_true - w
        loss = error**2
        gradient = -2 * error
        w_new = w - eta * gradient

        print(f"Step {step+1}:")
        print(f"   Guess: {w:.2f} | Error: {error:.2f} | Loss: {loss:.2f}")
        print(f"   Gradient: {gradient:.2f} → Update: {eta * gradient:.2f}")
        w = w_new

    print(f"\nFinal guess: {w:.2f} kg → Much closer!")


if __name__ == "__main__":
    gradient_descent_example()
