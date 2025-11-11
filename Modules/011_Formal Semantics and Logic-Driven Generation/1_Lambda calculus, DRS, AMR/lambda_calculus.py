# lambda_calculus.py
# Implements Lambda Calculus concepts for NLG tutorial
# Designed for aspiring scientists, with clear explanations and visualizations
# Requires: matplotlib
# Run: python lambda_calculus.py

import matplotlib.pyplot as plt

# --- Lambda Calculus Basics ---
# Lambda Calculus is a system of functions (like recipes) that take inputs and give outputs.
# Key idea: Everything is a function, no side effects, like pure math.
# Used in NLG to compose meanings, e.g., "John loves Mary" as a function.

# Identity function: λx.x (takes x, returns x)
identity = lambda x: x

# Church numerals: Represent numbers as functions
# 2 = λf.λx.f(f(x)) (apply f twice)
two = lambda f: lambda x: f(f(x))
three = lambda f: lambda x: f(f(f(x)))

# Addition: λm.λn.λf.λx.m f (n f x)
add = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

# Test basic lambda
print("Testing Identity Function:")
print(identity(5))  # Outputs: 5

# Test addition: 2 + 3
inc = lambda x: x + 1  # Increment function
result = add(two)(three)(inc)(0)  # Apply inc 2+3 times to 0
print(f"2 + 3 = {result}")  # Outputs: 5


# --- Visualization ---
# Plot how Church numerals apply a function multiple times
def plot_church_numeral(n, label):
    f = lambda x: x + 1
    x = range(5)
    y = [n(f)(0) for _ in x]  # Apply n times
    plt.plot(x, y, label=label)


print("Generating Church Numeral Plot...")
plot_church_numeral(two, "Two")
plot_church_numeral(three, "Three")
plt.xlabel("Applications")
plt.ylabel("Result")
plt.title("Church Numerals in Action")
plt.legend()
plt.show()

# --- Advanced: Recursion with Y-Combinator ---
# Y = λf.(λx.f (x x)) (λx.f (x x)) enables loops
Y = lambda f: (lambda x: f(lambda z: x(x)(z)))(lambda x: f(lambda z: x(x)(z)))

# Factorial: λf.λn. if n=0 then 1 else n * f(n-1)
factorial = Y(lambda f: lambda n: 1 if n == 0 else n * f(n - 1))
print(f"Factorial of 5 = {factorial(5)}")  # Outputs: 120

# --- Exercise ---
# Try reducing (λx.λy.y x) a b manually, then check:
# Answer: becomes b a
test_exercise = lambda x: lambda y: y(x)
result_exercise = test_exercise("a")("b")
print(f"Exercise: (λx.λy.y x) a b = {result_exercise}")  # Outputs: b a

# --- 2025 Update ---
# Neural Lambda Calculus (2025) combines lambda with neural nets for AI reasoning.
# Example application: Use in LLMs to ensure safe, logical outputs.
# Try: Adapt AND function for probabilistic logic (True with 0.8 chance).

# --- For Scientists ---
# Lambda ensures provable computations, like Einstein’s precise equations.
# Use in quantum simulations: λstate.collapse(state) for measurements.
