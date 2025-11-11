# basic_nlg.py
# Fundamentals of Natural Language Generation (NLG) using templates.
# As Turing would say, start with simple computable functions.

# Step 1: Define data (like structured input in research data)
data = {"temperature": 25, "condition": "sunny"}

# Step 2: Template generation (basic string mapping, akin to rule-based systems)
text = f"Today is a {data['condition']} day with temperature {data['temperature']}°C."
print(text)  # Output: Today is a sunny day with temperature 25°C.
# Explanation: Simple string formatting for basic NLG. In science, this turns data into readable reports.
